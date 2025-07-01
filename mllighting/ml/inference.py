import os

from PIL import Image

import torch
from torch import nn as torch_nn

from mllighting.ml import constants, dataset


def run_inference(
        model: torch_nn.Module,
        render_directory: str,
        image_size: tuple[int, int] = constants.IMAGE_SIZE,
        device: torch.device = torch.device('cpu')) -> list[float]:
    """Run the inference with the given model.

    Args:
        model: The model to run inference with.
        render_directory: The directory containing the images.
        image_size: The transform image size.
        device: The device to run the inference on.

    Returns:
        The infered values.
    """
    # Load the images.
    albedo = Image.open(
        os.path.join(
            render_directory,
            'albedo.png')).convert('RGB')
    beauty = Image.open(
        os.path.join(
            render_directory,
            'beauty.png')).convert('RGB')
    normal = dataset.read_exr_as_tensor(
        os.path.join(render_directory, 'normal.exr'), image_size=image_size)
    position = dataset.read_exr_as_tensor(
        os.path.join(render_directory, 'position.exr'), image_size=image_size)

    # Apply transforms.
    trans = dataset.get_transform(image_size=image_size)
    albedo = trans(albedo)
    beauty = trans(beauty)

    # Concatenate.
    image_tensor = torch.cat([
        beauty, albedo, normal, position
    ], dim=0)

    # Predict the values.
    inputs = image_tensor
    inputs = inputs.unsqueeze(0).to(device=device)
    with torch.no_grad():
        preds = model(inputs)
        predicted_lights = preds.squeeze(0).cpu().numpy()

    return predicted_lights.tolist()
