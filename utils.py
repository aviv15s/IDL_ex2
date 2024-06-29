from torch.utils.data import DataLoader, Subset
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from tqdm import tqdm


def get_dataloaders(train_size=None):
    """
    Returns a train and test dataloaders for MNIST digits dataset
    :param train_size: optional to limit size of train dataset
    :return: train_dataloader, test_dataloader
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transform
    )

    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transform
    )

    if train_size is not None:
        indices = torch.arange(100)
        training_data = Subset(training_data, indices)

    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    return train_dataloader, test_dataloader


def model_train_labels(model, dataloader, optimizer, criterion):
    """
    trains a model as a classifier
    :param model:
    :param dataloader:
    :param optimizer:
    :param criterion:
    :return:
    """
    model.train()
    total_loss = 0
    correct_count = 0
    for data in tqdm(dataloader):
        img, label = data
        onehot_label = torch.nn.functional.one_hot(label.to(torch.int64), 10)

        # Forward pass
        output = model(img)
        loss = criterion(output.to(torch.float64), onehot_label.to(torch.float64))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(img)

        predicted = torch.argmax(output, dim=1)
        correct_count += (predicted == label).sum().item()

    return total_loss / len(dataloader.dataset), 100 * correct_count / len(dataloader.dataset)

def model_test_labels(model, dataloader, loss_fn):
    """
    tests a model as a classifier
    :param model:
    :param dataloader:
    :param loss_fn:
    :return:
    """
    test_loss = 0
    correct_count = 0
    model.eval()

    with torch.no_grad():
        for data in dataloader:
            img, label = data
            output = model(img)
            test_loss += loss_fn(output, label).item()

            predicted = torch.argmax(output, dim=1)
            correct_count += (predicted == label).sum().item()
    return test_loss / len(dataloader), 100 * correct_count / len(dataloader.dataset)

def model_train_images(model, dataloader, optimizer, criterion):
    """
    trains a model as a reconstructor
    :param model:
    :param dataloader:
    :param optimizer:
    :param criterion:
    :return:
    """
    model.train()
    total_loss = 0
    for data in tqdm(dataloader):
        img, _ = data

        # Forward pass
        output = model(img)
        loss = criterion(output.to(torch.float64), img.to(torch.float64))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(img)
    return total_loss / len(dataloader.dataset)


def model_test_images(model, dataloader, loss_fn):
    """
    tests a model as a reconstructor
    :param model:
    :param dataloader:
    :param loss_fn:
    :return:
    """
    test_loss = 0
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            img, _ = data
            test_loss += loss_fn(model(img), img).item()

    test_loss /= len(dataloader)

    return test_loss


def plot_graphs_train_test_values(train_values, test_values, value_name, y_label, question):
    """
    plots graphs for train an test values

    :param train_values: list
        List of training  values per epoch.
    :param test_values: list
        List of test values per epoch.
    :param value_name: name of value plotted. used in titles
    :param y_label: label of y axis
    :return: None
    """
    plt.plot(list(range(1, len(train_values) + 1)), train_values, label=f'train {value_name}')
    plt.plot(list(range(1, len(test_values) + 1)), test_values, label=f'test {value_name}')
    plt.xlabel("# epoch")
    plt.ylabel(y_label)
    plt.title(f"Train and Test {value_name} per epoch Q{question}")
    plt.grid(True)
    plt.legend()
    plt.show()
