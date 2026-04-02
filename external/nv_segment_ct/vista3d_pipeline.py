import copy
import json
import logging
import os
import pathlib
from typing import Sequence

import numpy as np
import torch
from monai.apps.vista3d.transforms import VistaPostTransformd, VistaPreTransformd
from monai.data.utils import decollate_batch, list_data_collate
from monai.networks.utils import eval_mode, train_mode
from monai.transforms import (
    CastToTyped,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    Invertd,
    Lambdad,
    LoadImaged,
    Orientationd,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    reset_ops_id,
)
from monai.utils import ForwardMode, optional_import, set_determinism
from monai.utils.enums import CommonKeys as Keys
from monai.utils.module import look_up_option
from scripts.inferer import Vista3dInferer
from transformers import AutoModel, Pipeline
from transformers.pipelines import PIPELINE_REGISTRY

rearrange, _ = optional_import("einops", name="rearrange")

FILE_PATH = os.path.dirname(__file__)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class VISTA3DPipeline(Pipeline):
    """Define the VISTA3D pipeline."""

    PREPROCESSING_EXTRA_ARGS = [
        "image_key",
        "resample_spacing",
        "metadata_path",
        "load_image",
    ]
    INFERENCE_EXTRA_ARGS = [
        "mode",
        "amp",
        "hyper_kwargs",
        "roi_size",
        "overlap",
        "sw_batch_size",
        "use_point_window",
    ]
    POSTPROCESSING_EXTRA_ARGS = [
        "pred_key",
        "image_key",
        "output_dir",
        "output_ext",
        "output_postfix",
        "separate_folder",
        "save_output",
    ]
    EVERYTHING_LABEL = list(
        set([i + 1 for i in range(132)])
        - set([2, 16, 18, 20, 21, 23, 24, 25, 26, 27, 128, 129, 130, 131, 132])
    )

    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.preprocessing_transforms = self._init_preprocessing_transforms(
            **self._preprocess_params
        )
        self.inferer = self._init_inferer(**self._forward_params)
        self.postprocessing_transforms = self._init_postprocessing_transforms(
            **self._postprocess_params
        )

    def _init_inferer(
        self,
        roi_size: Sequence = (128, 128, 128),
        overlap: float = 0.3,
        sw_batch_size: int = 1,
        use_point_window: bool = True,
    ):
        return Vista3dInferer(
            roi_size=roi_size,
            overlap=overlap,
            use_point_window=use_point_window,
            sw_batch_size=sw_batch_size,
        )

    def _init_preprocessing_transforms(
        self,
        image_key: str = "image",
        resample_spacing: Sequence = (1.5, 1.5, 1.5),
        metadata_path: str = os.path.join(FILE_PATH, "metadata.json"),
        load_image: bool = True,
    ):
        device = self.device
        subclass = {
            "2": [14, 5],
            "20": [28, 29, 30, 31, 32],
            "21": list(range(33, 57)) + list(range(63, 98)) + [114, 120, 122],
        }
        metadata = json.loads(pathlib.Path(metadata_path).read_text())
        labels_dict = metadata["network_data_format"]["outputs"]["pred"]["channel_def"]
        preprocessing_list = [
                LoadImaged(keys=image_key, image_only=True),
                EnsureChannelFirstd(keys=image_key),
                EnsureTyped(keys=image_key, device=device, track_meta=True),
                Spacingd(keys=image_key, pixdim=resample_spacing, mode="bilinear"),
                CropForegroundd(
                    keys=image_key, allow_smaller=True, margin=10, source_key=image_key
                ),
                VistaPreTransformd(
                    keys=image_key, subclass=subclass, labels_dict=labels_dict
                ),
                ScaleIntensityRanged(
                    keys=image_key,
                    a_min=-963.8247715525971,
                    a_max=1053.678477684517,
                    b_min=0,
                    b_max=1,
                    clip=True,
                ),
                Orientationd(keys=image_key, axcodes="RAS"),
                CastToTyped(keys=image_key, dtype=torch.float32),
            ]
        if not load_image:
            preprocessing_list.pop(0)

        preprocessing_transforms = Compose(preprocessing_list)
        return preprocessing_transforms

    def _init_postprocessing_transforms(
        self,
        pred_key: str = "pred",
        image_key: str = "image",
        output_dir: str = "output_directory",
        output_ext: str = ".nii.gz",
        output_dtype: torch.dtype = torch.float32,
        output_postfix: str = "seg",
        separate_folder: bool = True,
        save_output: bool = True,
    ):
        transforms = [
            VistaPostTransformd(keys=pred_key),
            Invertd(
                keys=pred_key,
                transform=copy.deepcopy(self.preprocessing_transforms),
                orig_keys=image_key,
                nearest_interp=True,
                to_tensor=True,
            ),
            Lambdad(keys=pred_key, func=lambda x: torch.nan_to_num(x, nan=255)),
        ]
        if save_output:
            transforms.append(
                SaveImaged(
                    keys=pred_key,
                    resample=False,
                    output_dir=output_dir,
                    output_ext=output_ext,
                    output_dtype=output_dtype,
                    output_postfix=output_postfix,
                    separate_folder=separate_folder,
                ),
            )
        postprocessing_transforms = Compose(transforms=transforms)
        return postprocessing_transforms

    def _sanitize_parameters(self, **kwargs):
        """
        _sanitize_parameters exists to allow users to pass any parameters whenever they wish,
        be it at initialization time pipeline(...., maybe_arg=4) or at call time pipe = pipeline(...); output = pipe(...., maybe_arg=4).
        The returns of _sanitize_parameters are the 3 dicts of kwargs that will be passed directly to preprocess, _forward and postprocess.
        Don't fill anything if the caller didn't call with any extra parameter. That allows to keep the default arguments in the function
        definition which is always more “natural”."""

        vista3d_preprocessing_kwargs = {}
        vista3d_infer_kwargs = {}
        vista3d_postprocessing_kwargs = {}
        for key in self.INFERENCE_EXTRA_ARGS:
            if key in kwargs:
                vista3d_infer_kwargs[key] = kwargs[key]

        for key in self.PREPROCESSING_EXTRA_ARGS:
            if key in kwargs:
                vista3d_preprocessing_kwargs[key] = kwargs[key]

        for key in self.POSTPROCESSING_EXTRA_ARGS:
            if key in kwargs:
                vista3d_postprocessing_kwargs[key] = kwargs[key]

        return (
            vista3d_preprocessing_kwargs,
            vista3d_infer_kwargs,
            vista3d_postprocessing_kwargs,
        )

    def check_prompts_format(self, label_prompt, points, point_labels):
        """check the format of user prompts
        label_prompt: [1,2,3,4,...,B] List of tensors
        points: [[[x,y,z], [x,y,z], ...]] List of coordinates of a single object
        point_labels: [[1,1,0,...]] List of scalar that matches number of points
        """
        # check prompt is given
        if label_prompt is None and points is None:
            everything_labels = self.hyper_kwargs.get("everything_labels", None)
            if everything_labels is not None:
                label_prompt = [torch.tensor(_) for _ in everything_labels]
                return label_prompt, points, point_labels
            else:
                raise ValueError("Prompt must be given for inference.")
        # check label_prompt
        if label_prompt is not None:
            if isinstance(label_prompt, list):
                if not np.all([len(_) == 1 for _ in label_prompt]):
                    raise ValueError(
                        "Label prompt must be a list of single scalar, [1,2,3,4,...,]."
                    )
                if isinstance(label_prompt[0], list):
                    for prompt in label_prompt:
                        if not np.all([(x < 255).item() for x in prompt]):
                            raise ValueError(
                                "Current bundle only supports label prompt smaller than 255."
                            )
                else:
                    if not np.all([(x < 255).item() for x in label_prompt]):
                        raise ValueError(
                            "Current bundle only supports label prompt smaller than 255."
                        )
                if points is None:
                    supported_list = list(
                        {i + 1 for i in range(132)} - {16, 18, 129, 130, 131}
                    )
                    if isinstance(label_prompt[0], list):
                        for prompt in label_prompt:
                            if not np.all([(x < 255).item() for x in prompt]):
                                raise ValueError(
                                    "Current bundle only supports label prompt smaller than 255."
                                )
                    else:
                        if not np.all([x in supported_list for x in label_prompt]):
                            raise ValueError(
                                "Undefined label prompt detected. Provide point prompts for zero-shot."
                            )
            else:
                raise ValueError("Label prompt must be a list, [1,2,3,4,...,].")
        # check points
        if points is not None:
            if point_labels is None:
                raise ValueError("Point labels must be given if points are given.")
            if not np.all([len(_) == 3 for _ in points]):
                raise ValueError(
                    "Points must be three dimensional (x,y,z) in the shape of [[x,y,z],...,[x,y,z]]."
                )
            if len(points) != len(point_labels):
                raise ValueError("Points must match point labels.")
            if not np.all([_ in [-1, 0, 1, 2, 3] for _ in point_labels]):
                raise ValueError(
                    "Point labels can only be -1,0,1 and 2,3 for special flags."
                )
        if label_prompt is not None and points is not None:
            if len(label_prompt) != 1:
                raise ValueError(
                    "Label prompt can only be a single object if provided with point prompts."
                )
        # check point_labels
        if point_labels is not None:
            if points is None:
                raise ValueError("Points must be given if point labels are given.")
        return label_prompt, points, point_labels

    def transform_points(self, point, affine):
        """transform point to the coordinates of the transformed image
        point: numpy array [bs, N, 3]
        """
        bs, n = point.shape[:2]
        point = np.concatenate((point, np.ones((bs, n, 1))), axis=-1)
        point = rearrange(point, "b n d -> d (b n)")
        point = affine @ point
        point = rearrange(point, "d (b n)-> b n d", b=bs)[:, :, :3]
        return point

    def preprocess(
        self,
        inputs,
        **kwargs,
    ):
        for key, value in kwargs.items():
            if key in self._preprocess_params and value != self._preprocess_params[key]:
                logging.warning(
                    f"Please set the parameter {key} during initialization."
                )

            if key not in self.PREPROCESSING_EXTRA_ARGS:
                logging.warning(f"Cannot set parameter {key} for preprocessing.")
        inputs = self.preprocessing_transforms(inputs)
        inputs = list_data_collate([inputs])
        return inputs

    def _forward(
        self,
        inputs,
        mode: str = ForwardMode.EVAL,
        amp: bool = True,
        hyper_kwargs: dict = {"user_prompt": 1, "everything_labels": 1},
    ):
        set_determinism(seed=123)

        if inputs is None:
            raise ValueError("Must provide input data for inference.")
        self.hyper_kwargs = hyper_kwargs

        label_set = hyper_kwargs.get("label_set", None)
        # this validation label set should be consistent with 'labels.unique()', used to generate fg/bg points
        val_label_set = hyper_kwargs.get("val_label_set", label_set)
        # If user provide prompts in the inference, input image must contain original affine.
        # the point coordinates are from the original_affine space, while image here is after preprocess transforms.
        if hyper_kwargs["user_prompt"]:
            inputs, label_prompt, points, point_labels = (
                inputs["image"],
                inputs.get("label_prompt", None),
                inputs.get("points", None),
                inputs.get("point_labels", None),
            )
            labels = None
            label_prompt, points, point_labels = self.check_prompts_format(
                label_prompt, points, point_labels
            )
            inputs = inputs.to(self.device)
            # For N foreground object, label_prompt is [1, N], but the batch number 1 needs to be removed. Convert to [N, 1]
            label_prompt = (
                torch.as_tensor([label_prompt]).to(inputs.device)[0].unsqueeze(-1)
                if label_prompt is not None
                else None
            )
            # For points, the size can only be [1, K, 3], where K is the number of points for this single foreground object.
            if points is not None:
                points = torch.as_tensor([points])
                points = self.transform_points(
                    points,
                    np.linalg.inv(inputs.affine[0])
                    @ inputs.meta["original_affine"][0].numpy(),
                )
                points = torch.from_numpy(points).to(inputs.device)
            point_labels = (
                torch.as_tensor([point_labels]).to(inputs.device)
                if point_labels is not None
                else None
            )

        # If validation with ground truth label available.
        else:
            # TODO add these as attribute.
            inputs, labels = inputs["image"], inputs["label"]
            # create label prompt, this should be consistent with the label prompt used for training.
            if label_set is None:
                output_classes = hyper_kwargs.get("output_classes", None)
                label_set = np.arange(output_classes).tolist()
            label_prompt = torch.tensor(label_set).to(self.device).unsqueeze(-1)
            # point prompt is generated withing vista3d, provide empty points
            points = torch.zeros(label_prompt.shape[0], 1, 3).to(inputs.device)
            point_labels = -1 + torch.zeros(label_prompt.shape[0], 1).to(inputs.device)
            # validation for either auto or point.
            if hyper_kwargs.get("val_head", "auto") == "auto":
                # automatic only validation
                # remove val_label_set, vista3d will not sample points from gt labels.
                val_label_set = None
            else:
                # point only validation
                label_prompt = None

        # put iteration outputs into outputs TODO need to align with the customized inputs
        outputs = {Keys.IMAGE: inputs, Keys.LABEL: labels}
        mode = look_up_option(mode, ForwardMode)
        if mode == ForwardMode.EVAL:
            mode = eval_mode
        elif mode == ForwardMode.TRAIN:
            mode = train_mode
        else:
            raise ValueError(f"unsupported mode: {mode}, should be 'eval' or 'train'.")

        # execute forward computation
        self.model.network.to(self.device)
        with mode(self.model):
            if amp:
                with torch.autocast("cuda"):
                    outputs[Keys.PRED] = self.inferer(
                        inputs=inputs,
                        network=self.model.network,
                        point_coords=points,
                        point_labels=point_labels,
                        class_vector=label_prompt,
                        labels=labels,
                        label_set=val_label_set,
                    )
            else:
                outputs[Keys.PRED] = self.inferer(
                    inputs=inputs,
                    network=self.model.network,
                    point_coords=points,
                    point_labels=point_labels,
                    class_vector=label_prompt,
                    labels=labels,
                    label_set=val_label_set,
                )
        inputs = reset_ops_id(inputs)
        # Add dim 0 for decollate batch
        outputs["label_prompt"] = (
            label_prompt.unsqueeze(0) if label_prompt is not None else None
        )
        outputs["points"] = points.unsqueeze(0) if points is not None else None
        outputs["point_labels"] = (
            point_labels.unsqueeze(0) if point_labels is not None else None
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return outputs

    def postprocess(self, outputs, **kwargs):
        outputs[Keys.IMAGE] = outputs[Keys.IMAGE].to(self.device)
        outputs[Keys.PRED] = outputs[Keys.PRED].to(self.device)
        for key, value in kwargs.items():
            if key not in self.POSTPROCESSING_EXTRA_ARGS:
                logging.warning(f"Cannot set parameter {key} for postprocessing.")
            if (
                key in self._postprocess_params
                and value != self._postprocess_params[key]
            ) or (key not in self._postprocess_params):
                self._postprocess_params.update(kwargs)
                self.postprocessing_transforms = self._init_postprocessing_transforms(
                    **self._postprocess_params
                )

        outputs = self.postprocessing_transforms(decollate_batch(outputs))
        return outputs


def register_simple_pipeline():
    PIPELINE_REGISTRY.register_pipeline(
        "vista3d",
        pipeline_class=VISTA3DPipeline,
        pt_model=AutoModel,
        default={"pt": (os.path.join(FILE_PATH, "vista3d_pretrained_model"), "")},
        type="image",  # current support type: text, audio, image, multimodal
    )
