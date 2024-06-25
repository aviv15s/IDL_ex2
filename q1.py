import data_loaders as dl
import torch.nn as nn
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import networks

def model_train(model, dataloader, optimizer, criterion):
    best_loss = 1000000
    model.train()
    epoch_num = 30
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
        best_loss = min(best_loss, loss.item())
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
    model = networks.AutoEncoder(12)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_loss = model_train(model, train_dataloader, optimizer, criterion)
    model_test(model, test_dataloader)
    torch.save(model.state_dict(), f'q1_model_{round(train_loss,2)}.pth')



if __name__ == "__main__":
    run_q1()
