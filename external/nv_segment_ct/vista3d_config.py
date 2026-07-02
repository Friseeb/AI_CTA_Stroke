from transformers import PretrainedConfig


class VISTA3DConfig(PretrainedConfig):
    """Configuration class for vista3d"""

    model_type = "VISTA3D"

    def __init__(self, encoder_embed_dim: int = 48, input_channels: int = 1, **kwargs):
        """
        Set the hyperparameters for the VISTA3D model.

        Parameters:
            input_channels: channel of input images.
            encoder_embed_dim: the encoder_embed_dim of the VISTA3D model.
        """
        self.input_channels = input_channels
        self.encoder_embed_dim = encoder_embed_dim
        super().__init__(**kwargs)
