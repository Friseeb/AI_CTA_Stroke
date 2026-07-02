from transformers import pipeline
from vista3d_config import VISTA3DConfig
from vista3d_model import VISTA3DModel, register_my_model
from vista3d_pipeline import VISTA3DPipeline, register_simple_pipeline


class HuggingFacePipelineHelper:

    def __init__(self, pipeline_name: str = "vista3d"):
        self.pipeline_name = pipeline_name

    def __model_register(self):
        register_my_model()

    def __pipeline_register(self):
        register_simple_pipeline()

    def get_pipeline(self):
        self.__model_register()
        self.__pipeline_register()
        return pipeline(self.pipeline_name)

    def _update_config(self, config, config_dict):
        if config_dict:
            for key in config_dict:
                if hasattr(config, key) and getattr(config, key) != config_dict[key]:
                    setattr(config, key, config_dict[key])
        return config

    def init_pipeline(self, pretrained_model_name_or_path: str, **kwargs):
        config = VISTA3DConfig()
        config_dict = kwargs.pop("config_dict", None)
        self._update_config(config, config_dict)
        model = VISTA3DModel(config)
        model = model.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path
        )
        return VISTA3DPipeline(model, **kwargs)
