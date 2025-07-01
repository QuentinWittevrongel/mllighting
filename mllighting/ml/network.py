import torch
import torch.nn as torch_nn

from mllighting.ml import constants


class CNNModel(torch_nn.Module):

    def __init__(
            self,
            image_size: tuple[int, int] = constants.IMAGE_SIZE,
            input_channels: int = 12):
        super().__init__()

        input_pixelcount = image_size[0] * image_size[1]

        self.conv_layers = torch_nn.Sequential(
            torch_nn.Conv2d(input_channels, 8, 3, stride=1, padding=1),
            torch_nn.ReLU(),
            torch_nn.MaxPool2d(2, stride=2),
            torch_nn.Conv2d(8, 16, 3, stride=1, padding=1),
            torch_nn.ReLU(),
            torch_nn.MaxPool2d(2, stride=2),
        )

        self.flatten_layers = torch_nn.Sequential(
            torch_nn.Flatten(),
            torch_nn.Linear(input_pixelcount, 128),
            torch_nn.ReLU(),
            torch_nn.Linear(128, 3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.flatten_layers(x)
        return x


def load_model(
        checkpoint: str | None = None,
        device: torch.device = torch.device('cpu')) -> CNNModel:
    """Load the model with an optional checkpoint.

    Args:
        checkpoint: The model checkpoint to load.
        device: The device to load the model with.

    Returns:
        The model.
    """
    model = CNNModel().to(device=device)
    if checkpoint is not None:
        state_dict = torch.load(
            checkpoint,
            weights_only=True,
            map_location=device)
        model.load_state_dict(state_dict)
    return model
