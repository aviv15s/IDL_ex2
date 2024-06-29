import torch
import torch.nn as nn
import torch.optim as optim
from utils import get_dataloaders, model_train_images, model_test_images, plot_graphs_train_test_values
import networks


def run_q1(epochs):
    train_dataloader, test_dataloader = get_dataloaders()
    model = networks.AutoEncoder(12)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_loss_list, test_loss_list = [], []
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loss = model_train_images(model, train_dataloader, optimizer, criterion)
        test_loss = model_test_images(model, test_dataloader, criterion)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        print(f'Train loss: {train_loss}, Test loss: {test_loss}')

    torch.save(model.state_dict(), f'q1_model.pth')
    torch.save(model.encoder.state_dict(), f'q1_encoder.pth')

    plot_graphs_train_test_values(train_loss_list, test_loss_list, "loss", "loss [arb]", 1)


if __name__ == "__main__":
    run_q1(30)
