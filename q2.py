import torch
import torch.nn as nn
import torch.optim as optim
from utils import get_dataloaders, plot_graphs_train_test_values, model_train_labels, model_test_labels
import networks


def run_q2(epochs):
    train_dataloader, test_dataloader = get_dataloaders()
    model = networks.Classifier(10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_loss_list, test_loss_list = [], []
    train_accuracy_list, test_accuracy_list = [], []
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loss, train_accuracy = model_train_labels(model, train_dataloader, optimizer, criterion)
        test_loss, test_accuracy = model_test_labels(model, test_dataloader, criterion)
        train_loss_list.append(train_loss)
        train_accuracy_list.append(train_accuracy)
        test_loss_list.append(test_loss)
        test_accuracy_list.append(test_accuracy)
        print(f'Train loss: {train_loss}, Test loss: {test_loss}')
        print(f'Train Accuracy: {round(train_accuracy, 3)}%, Test Accuracy: {round(test_accuracy, 3)}%')

    torch.save(model.state_dict(), f'q2_full_model.pth')
    torch.save(model.encoder.state_dict(), f'q2_encoder_model.pth')

    plot_graphs_train_test_values(train_loss_list, test_loss_list, "loss", "loss [arb]", 2)
    plot_graphs_train_test_values(train_accuracy_list, test_accuracy_list, "accuracy", "%", 2)


if __name__ == "__main__":
    run_q2(30)
