import argparse

import torch

from mllighting.ml import network, train


def main(args: argparse.Namespace):
    # Detect the device to use.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    # Initialize the model.
    model = network.CNNModel().to(device=device)

    # Load the checkpoint path.
    checkpoint = torch.load(args.checkpoint, weights_only=True)
    model.load_state_dict(checkpoint)

    # Test the model.
    result = train.test_model(
        model,
        args.directory,
        device=device)
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='ML Lighting testing script',
        description='Test script for the ML Lighting tool')

    parser.add_argument('directory', help='The dataset directory')
    parser.add_argument('checkpoint', help='The checkpoint to use')

    args = parser.parse_args()

    main(args)
