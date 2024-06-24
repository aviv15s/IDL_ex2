import data_loaders as dl
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 5, stride=2)  # [batch, 8, 12, 12]
        self.conv2 = nn.Conv2d(8, 16, 5, stride=2)  # [batch, 16, 4, 4]
        self.conv3 = nn.Conv2d(16, 32, 3)  # [batch, 32, 2, 2]
        self.fc = nn.Linear(32*4, 12)  # Reduce to 12 dimensions

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(12, 32*4)  # Increase dimensions from 12 to 64
        self.conv1 = nn.ConvTranspose2d(32, 16, 3)  # [batch, 32, 7, 7]
        self.conv2 = nn.ConvTranspose2d(16, 8, 5, stride=2, output_padding=1)  # [batch, 16, 14, 14]
        self.conv3 = nn.ConvTranspose2d(8, 1, 5, stride=2, output_padding=1)  # [batch, 1, 28, 28]

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 32, 2, 2)  # Reshape to [batch, 32, 1, 1]
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.tanh(self.conv3(x))  # Output normalized to [-1, 1]
        return x


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x





def train_model(model, dataloader, optimizer, criterion):
    model.train()
    epoch_num = 10
    for epoch in range(epoch_num):
        for data in dataloader:
            img, _ = data

            # Forward pass
            output = model(img)
            loss = criterion(output, img)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{epoch_num}], Loss: {loss.item():.4f}')


def eval_model(model, dataloader):
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            img, _ = data
            output = model(img)

            # Plot original images
            plt.figure(figsize=(9, 2))
            for i in range(6):
                plt.subplot(2, 6, i + 1)
                plt.imshow(img[i][0], cmap='gray')
                plt.axis('off')

            # Plot reconstructed images
            for i in range(6):
                plt.subplot(2, 6, i + 7)
                plt.imshow(output[i][0], cmap='gray')
                plt.axis('off')

            plt.show()
            break



def run_q1():
    train_dataloader, test_dataloader = dl.get_dataloaders()

    model = AutoEncoder()
    criterion = nn.L1Loss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    train_model(model, train_dataloader, optimizer, criterion)
    eval_model(model, test_dataloader)


if __name__ == "__main__":
    run_q1()
