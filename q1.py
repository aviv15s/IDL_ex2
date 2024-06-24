import data_loaders as dl
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # self.conv1 = nn.Conv2d(1, 4, 5, stride=2)  # [batch, 8, 12, 12]
        # self.conv2 = nn.Conv2d(4, 8, 3)  # [batch, 8, 10, 10]
        # self.conv3 = nn.Conv2d(8, 16, 3, stride=2)  # [batch, 16, 8, 8]
        # self.conv4 = nn.Conv2d(16, 32, 3)  # [batch, 16, 8, 8]
        # # self.conv5 = nn.Conv2d(32, 32, 3)  # [batch, 32, 2, 2]
        # self.fc = nn.Linear(32*4, 12)  # Reduce to 12 dimensions

        self.conv1 = nn.Conv2d(1, 4, 5)  # [batch, 8, 24, 24]
        self.conv2 = nn.Conv2d(4, 8, 3, stride=2)  # [batch, 8, 11, 11]
        self.conv3 = nn.Conv2d(8, 16, 2, stride=2)  # [batch, 16, 10, 10]
        self.conv4 = nn.Conv2d(16, 32, 3, stride=2)  # [batch, 16, 8, 8]
        self.fc = nn.Linear(32*4, 12)  # Reduce to 12 dimensions


    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc(x))
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(12, 32 * 2 * 2)  # Increase dimensions from 12 to 128
        self.conv1 = nn.ConvTranspose2d(32, 16, 3)  # [batch, 16, 4, 4]
        self.conv2 = nn.ConvTranspose2d(16, 8, 3, stride=2, output_padding=1)  # [batch, 8, 10, 10]
        self.conv3 = nn.ConvTranspose2d(8, 4, 3)  # [batch, 4, 12, 12]
        self.conv4 = nn.ConvTranspose2d(4, 1, 5, stride=2, output_padding=1)  # [batch, 1, 28, 28]

        # self.fc = nn.Linear(12, 32 * 2 * 2)  # Increase dimensions from 12 to 128
        # self.conv1 = nn.ConvTranspose2d(32, 16, 3)  # [batch, 16, 4, 4]
        # self.conv2 = nn.ConvTranspose2d(16, 8, 5, stride=2, output_padding=1)  # [batch, 8, 10, 10]
        # self.conv3 = nn.ConvTranspose2d(8, 1, 5, stride=2, output_padding=1)  # [batch, 1, 28, 28]

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 32, 2, 2)  # Reshape to [batch, 32, 2, 2]
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        # x = torch.sigmoid(self.conv3(x))
        x = torch.relu(self.conv3(x))
        x = self.conv4(x)  # Output normalized to [0, 1]
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





def model_train(model, dataloader, optimizer, criterion):
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


def model_test(model, dataloader):
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            img, _ = data
            output = model(img)

            # Plot original images
            plt.figure(figsize=(12, 4))
            for i in range(6):
                plt.subplot(2, 6, i + 1)
                plt.imshow(img[i][0].cpu().numpy(), cmap='gray')
                plt.title('Original')
                plt.axis('off')

            # Plot reconstructed images
            for i in range(6):
                plt.subplot(2, 6, i + 7)
                plt.imshow(output[i][0].cpu().numpy(), cmap='gray')
                plt.title('Reconstructed')
                plt.axis('off')

            plt.show()
            break



def run_q1():
    train_dataloader, test_dataloader = dl.get_dataloaders()
    model = AutoEncoder()
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model_train(model, train_dataloader, optimizer, criterion)
    model_test(model, test_dataloader)


if __name__ == "__main__":
    run_q1()
