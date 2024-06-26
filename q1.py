import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import data_loaders as dl
import networks


def model_train(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in tqdm(dataloader):
        img, _ = data
        # label = torch.nn.functional.one_hot(label.to(torch.int64), 10)
        # Forward pass
        output = model(img)
        loss = criterion(output.to(torch.float64), img.to(torch.float64))
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def model_test(model, dataloader, loss_fn):
    test_loss = 0
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            img, _ = data
            test_loss += loss_fn(model(img), img).item()
            # Plot original images
            #             plt.figure(figsize=(12, 4))
            #             for i in range(6):
            #                 plt.subplot(2, 6, i + 1)
            #                 plt.imshow(img[i][0].cpu().numpy(), cmap='gray')
            #                 plt.title('Original')
            #                 plt.axis('off')
            #
            #             # Plot reconstructed images
            #             for i in range(6):
            #                 plt.subplot(2, 6, i + 7)
            #                 plt.imshow(output[i][0].cpu().numpy(), cmap='gray')
            #                 plt.title('Reconstructed')
            #                 plt.axis('off')
            #
            #             plt.show()
            #             break
    test_loss /= len(dataloader)

    return test_loss


def plot_epochs_loss(train_loss_list, test_loss_list):
    """
    Plot the training and test loss per epoch.

    :param train_loss_list: list
        List of training loss values per epoch.
    :param test_loss_list: list
        List of test loss values per epoch.
    :return: None
    """
    plt.plot(list(range(1, len(train_loss_list) + 1)), train_loss_list, label='train loss')
    plt.plot(list(range(1, len(test_loss_list) + 1)), test_loss_list, label='test loss')
    plt.xlabel("# epoch")
    plt.ylabel('loss [arb]')
    plt.title("Train and Test loss per epoch")
    plt.grid(True)
    plt.legend()
    plt.show()


def run_q1():
    train_dataloader, test_dataloader = dl.get_dataloaders()
    model = networks.AutoEncoder(12)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 30
    train_loss_list, test_loss_list = [], []
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loss = model_train(model, train_dataloader, optimizer, criterion)
        test_loss = model_test(model, test_dataloader, criterion)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        print(f'Train loss: {train_loss}, Test loss: {test_loss}')
    torch.save(model.state_dict(), f'q1_model.pth')
    plot_epochs_loss(train_loss_list, test_loss_list)


if __name__ == "__main__":
    run_q1()
