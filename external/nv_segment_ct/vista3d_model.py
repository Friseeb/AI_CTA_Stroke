import os

import monai.networks.nets
import torch
from transformers import AutoConfig, AutoModel, PreTrainedModel
from vista3d_config import VISTA3DConfig


class VISTA3DModel(PreTrainedModel):
    """VISTA3D model for hugging face"""

    config_class = VISTA3DConfig

    def __init__(self, config):
        super().__init__(config)
        if config.model_type == "VISTA3D":
            self.network = monai.networks.nets.vista3d132(
                encoder_embed_dim=config.encoder_embed_dim,
                in_channels=config.input_channels,
            )

    def forward(self, input):
        return self.network(input)


def register_my_model():
    """Utility function to register VISTA3D model so that it can be instantiate by the AutoModel function."""
    AutoConfig.register("VISTA3D", VISTA3DConfig)
    AutoModel.register(VISTA3DConfig, VISTA3DModel)


if __name__ == "__main__":
    FILE_PATH = os.path.dirname(__file__)
    MODEL_WEIGHT_PATH = os.path.join(FILE_PATH, "models/model.pt")
    MODEL_PATH = os.path.join(FILE_PATH, "vista3d_pretrained_model")
    config = VISTA3DConfig()
    hugging_face_model = VISTA3DModel(config)
    hugging_face_model.network.load_state_dict(torch.load(MODEL_WEIGHT_PATH))
    hugging_face_model.save_pretrained(MODEL_PATH)
