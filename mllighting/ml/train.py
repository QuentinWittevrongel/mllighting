import copy

import torch
import torch.nn as torch_nn
import torch.utils.data as torch_data
import torch.optim.optimizer as torch_optimizer

from mllighting.ml import constants, dataset


def get_loss_function() -> torch_nn.Module:
    """Get the loss function to use.

    Returns:
        The loss function.
    """
    return torch_nn.MSELoss()


def train_loop(
        model: torch_nn.Module,
        loader: torch_data.DataLoader,
        criterion: torch_nn.Module,
        optimizer: torch_optimizer.Optimizer,
        num_epochs: int = constants.EPOCH_COUNT,
        device: torch.device = torch.device('cpu')) -> torch_nn.Module:
    """Train the model for a number of epoch and returns the best version.

    Args:
        model: The model to train.
        loader: The loader of the data to train on.
        criterion: The loss funtion to use for the training.
        optimizer: The optimizer function to use for the training.
        num_epochs: The number of epoch to train the model.
        device: The device to run the train on.

    Returns:
        The best trained model.
    """
    lowest_score = 1e10
    best_model = None

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')

        model.train()
        for inputs, targets in loader:
            # Load the data.
            inputs = inputs.to(device=device)
            targets = targets.to(device=device)

            # Predict.
            preds = model(inputs)
            loss = criterion(preds, targets)

            # Compute the gradients.
            optimizer.zero_grad()
            loss.backward()

            # Update the model.
            optimizer.step()

        if loss < lowest_score:
            print(f'New lowest: {loss}')
            lowest_score = loss

            best_model = copy.deepcopy(model)

            # path = os.path.join(output_directory, f'model_{epoch}.pth')
            # torch.save(model.state_dict(), path)

            # best_state = model.state_dict()

        print(f'Loss: {loss}')

    if best_model is None:
        raise ValueError('No best model found. This should not happen')

    return best_model


def train_model(
        model: torch_nn.Module,
        dataset_directory: str,
        num_epochs: int = constants.EPOCH_COUNT,
        device: torch.device = torch.device('cpu')) -> torch_nn.Module:
    """Train the model for a number of epoch and returns the best version.

    Args:
        model: The model to train.
        dataset_directory: The dataset directory.
        device: The device to run the train on.

    Returns:
        The best trained model.
    """
    lossfunc = get_loss_function()
    # Define optimiser.
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Load the data.
    train_dataset = dataset.RenderMapsDataset(dataset_directory)
    train_loader = torch_data.DataLoader(
        train_dataset, batch_size=constants.BATCH_SIZE, shuffle=True)

    best_model = train_loop(
        model,
        train_loader,
        lossfunc,
        optimizer,
        num_epochs=num_epochs,
        device=device)

    return best_model


def test_model(
        model: torch_nn.Module,
        dataset_directory: str,
        device: torch.device = torch.device('cpu')) -> float:
    """Test the given model on the specified data set.

    Args:
        model: The model to test.
        dataset_directory: The dataset to use as test.
        device: The device to run the test on.

    Returns:
        The average loss value.
    """
    lossfunc = get_loss_function()

    # Load the data.
    test_dataset = dataset.RenderMapsDataset(dataset_directory)
    test_loader = torch_data.DataLoader(
        test_dataset, shuffle=True)

    # Run the model on the data set and get the average loss.
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device=device)
            targets = targets.to(device=device)

            preds = model(inputs)
            loss = lossfunc(preds, targets)
            running_loss += loss

    return running_loss/len(test_loader)
