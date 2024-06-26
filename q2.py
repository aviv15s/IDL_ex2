
import torch
import torch.nn as nn
import torch.optim as optim

import data_loaders as dl
import networks
from tqdm import tqdm
from q1 import plot_epochs_loss as plot_epochs_loss

def model_train(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in tqdm(dataloader):
        img, label = data
        label = torch.nn.functional.one_hot(label.to(torch.int64), 10)
        # Forward pass
        output = model(img)
        loss = criterion(output.to(torch.float64), label.to(torch.float64))
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
            img, label = data
            test_loss += loss_fn(model(img), label).item()

    test_loss /= len(dataloader)
    return test_loss


def run_q2():
    train_dataloader, test_dataloader = dl.get_dataloaders()
    model = networks.Classifier(10)
    criterion = nn.CrossEntropyLoss()
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

    torch.save(model.state_dict(), f'q2_full_model.pth')
    torch.save(model.encoder.state_dict(), f'q2_encoder_model.pth')

    plot_epochs_loss(train_loss_list, test_loss_list)



if __name__ == "__main__":
    run_q2()
