import argparse

import torch

from mllighting.ml import network, train


def main(args: argparse.Namespace):
    # Detect the device to use.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    # Initialize the model.
    model = network.CNNModel().to(device=device)

    # Train the model.
    best_model = train.train_model(
        model,
        args.directory,
        device=device)

    # Save the best model trained.
    torch.save(best_model.state_dict(), args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='ML Lighting training script',
        description='Training script for the ML Lighting tool')

    parser.add_argument('directory', help='The dataset directory')
    parser.add_argument('output', help='The checkout output')

    args = parser.parse_args()

    main(args)
