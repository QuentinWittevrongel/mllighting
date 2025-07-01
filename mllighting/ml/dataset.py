import json
import os

import numpy

import OpenImageIO

from PIL import Image

import torch
import torch.utils.data as torch_data

import torchvision.transforms as transforms

from mllighting.ml import constants


class RenderMapsDataset(torch_data.Dataset):
    """The render map dataset

    Data is stored under the dataset directory with the following structure:

    ```
    |- dataset directory
       |- sample index
          |- albedo.png
          |- beauty.png
          |- normal.exr
          |- position.exr
          |- light.json
    ```

    with the sample index starting at 0.

    The light json contains the lights information to train the model with.

    ```
    [
        {
            matrix: [16 floats]
        }
    ]
    ```
    """

    def __init__(
            self,
            directory: str,
            image_size: tuple[int, int] = constants.IMAGE_SIZE):
        """Initialize the dataset.

        Args:
            directory: The dataset directory.
            image_size: The image size to work with.
        """
        self.image_size = image_size
        self.directory = directory
        self.transform = get_transform(image_size=image_size)

    def __len__(self) -> int:
        # Get the number of directory in the dataset directory.
        count = len(
            [item for item in os.listdir(self.directory)
             if os.path.isdir(os.path.join(self.directory, item))])
        return count

    def __getitem__(self, index: int) -> dict:
        sample_directory = os.path.join(self.directory, str(index))

        # Load the render maps.
        albedo = Image.open(
            os.path.join(
                sample_directory,
                'albedo.png')).convert('RGB')
        beauty = Image.open(
            os.path.join(
                sample_directory,
                'beauty.png')).convert('RGB')
        normal = read_exr_as_tensor(
            os.path.join(sample_directory, 'normal.exr'),
            image_size=self.image_size)
        position = read_exr_as_tensor(
            os.path.join(sample_directory, 'position.exr'),
            image_size=self.image_size)

        # Apply transforms on the render maps.
        albedo = self.transform(albedo)
        beauty = self.transform(beauty)

        # The beauty is supposed to be drawn by the user.
        # Add variation to the beauty to compress the shadows and lighted
        # areas to simulate harder brush strokes.
        if torch.rand(1).item() < 0.5:
            gamma = torch.empty(1).uniform_(0.4, 0.8).item()
            beauty = (beauty + 1.0) * 0.5
            beauty = torch.clamp(beauty, 0.0, 1.0)
            beauty = beauty.pow(gamma)
            beauty = beauty * 2.0 - 1.0

        # Concatenate into a single tensor.
        image_tensor = torch.cat([
            beauty, albedo, normal, position
        ], dim=0)

        # Load light positions from the json file.
        light_filepath = os.path.join(sample_directory, 'light.json')
        with open(light_filepath, 'r') as f:
            lights_data = json.load(f)

        # Extract light positions as a tensor.
        lights_transforms = []
        for light_dict in lights_data:
            matrix = light_dict['matrix']
            lights_transforms.extend(
                [matrix[12], matrix[13], matrix[14]])

        light_tensor = torch.tensor(
            lights_transforms, dtype=torch.float32)

        return image_tensor, light_tensor


def get_transform(
        image_size: tuple[int, int] = constants.IMAGE_SIZE)\
        -> transforms.Compose:
    """Get the transform used by the data set.

    Args:
        image_size: The size to use for the images.

    Returns:
        The transform.
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def read_exr_as_tensor(
        filepath: str,
        image_size: tuple[int, int] = constants.IMAGE_SIZE) -> torch.Tensor:
    """Read and resze an EXR file and return a torch tensor.

    Args:
        filepath: The EXR file path.
        image_size: The size used to resize the EXR.

    Returns:
        The tensor.
    """
    # Load the exr file.
    src_buf = OpenImageIO.ImageBuf(filepath)
    spec = src_buf.spec()

    # Create a destination buffer with the same properties than the source
    # buffer, but resized.
    dst_buf = OpenImageIO.ImageBuf(
        OpenImageIO.ImageSpec(
            image_size[0], image_size[1], spec.nchannels, OpenImageIO.FLOAT))
    OpenImageIO.ImageBufAlgo.resize(dst_buf, src_buf)
    data = dst_buf.get_pixels(OpenImageIO.FLOAT)
    data = numpy.asarray(data).reshape(
        image_size[1], image_size[0], spec.nchannels)
    # Convert the data to tensor and match the order from PIL.
    tensor = torch.from_numpy(data).permute(2, 0, 1).float()
    return tensor
