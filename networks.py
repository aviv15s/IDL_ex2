import torch.nn as nn
import torch



class Encoder(nn.Module):
    def __init__(self, d):
        super(Encoder, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(1, 4, 5, stride=2),  # [batch, 4, 12, 12]
            nn.ReLU(),
            nn.Conv2d(4, 8, 3),  # [batch, 8, 10, 10]
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2),  # [batch, 16, 4, 4]
            nn.ReLU(),
            nn.Conv2d(16, 32, 3),  # [batch, 32, 2, 2]
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 4, d),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class Decoder(nn.Module):
    def __init__(self, d):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(d, 32 * 2 * 2)  # Increase dimensions from 12 to 128
        self.conv1 = nn.ConvTranspose2d(32, 16, 3)  # [batch, 16, 4, 4]
        self.conv2 = nn.ConvTranspose2d(16, 8, 3, stride=2, output_padding=1)  # [batch, 8, 10, 10]
        self.conv3 = nn.ConvTranspose2d(8, 4, 3)  # [batch, 4, 12, 12]
        self.conv4 = nn.ConvTranspose2d(4, 1, 5, stride=2, output_padding=1)  # [batch, 1, 28, 28]

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
    def __init__(self, d):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(d)
        self.decoder = Decoder(d)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
