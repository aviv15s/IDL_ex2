import q1
import torch.nn as nn
import torch
import data_loaders as dl
import torch.optim as optim
import matplotlib.pyplot as plt
import networks





def model_train(model, dataloader, optimizer, criterion):
    best_loss = 10000
    model.train()
    epoch_num = 10
    for epoch in range(epoch_num):
        for data in dataloader:
            img, label = data
            label = torch.nn.functional.one_hot(label.to(torch.int64), 10)

            # Forward pass
            output = model(img)
            loss = criterion(output.to(torch.float64), label.to(torch.float64))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            best_loss = min(best_loss, loss.item())
        print(f'Epoch [{epoch + 1}/{epoch_num}], Loss: {loss.item():.4f}')
    return best_loss


def model_test(model, dataloader):
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            img, label = data
            predictions = torch.argmax(model(img), 1)

            # Plot original images
            plt.figure(figsize=(12, 4))
            for i in range(6):
                plt.subplot(2, 6, i + 1)
                plt.imshow(img[i][0].cpu().numpy(), cmap='gray')
                plt.title(f'Prediction: {predictions[i]}')
                plt.axis('off')

            plt.show()
            break


def run_q2():
    train_dataloader, test_dataloader = dl.get_dataloaders()
    model = networks.Classifier(10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_loss = model_train(model, train_dataloader, optimizer, criterion)
    model_test(model, test_dataloader)
    torch.save(model.state_dict(), f'q2_model_{round(train_loss, 2)}.pth')


if __name__ == "__main__":
    run_q2()
