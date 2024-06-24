import q1
import torch.nn as nn
import torch
import data_loaders as dl
import torch.optim as optim
import matplotlib.pyplot as plt
class Classifier(q1.Encoder):
    def __init__(self):
        super(Classifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 4, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(4, 8, 3),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.Linear(32 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 20),
            nn.ReLU(),
            nn.Linear(20,10),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layers(x)
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



def run_q2():
    train_dataloader, test_dataloader = dl.get_dataloaders()
    model = Classifier()
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model_train(model, train_dataloader, optimizer, criterion)
    model_test(model, test_dataloader)


if __name__ == "__main__":
    run_q2()

