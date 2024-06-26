import torch
from matplotlib import pyplot as plt
from torch import nn, optim

import data_loaders as dl
import networks
from q1 import model_train as model_train, model_test as model_test, plot_epochs_loss as plot_epochs_loss


def plot_results(classifier_model, ae_model, dataloader):
    num_images = 50
    classifier_model.eval()
    ae_model.eval()
    # decoder_model.eval()

    with torch.no_grad():
        for data in dataloader:  # TODO select single batch
            img, label = data
            ae_reconstruction = ae_model(img)
            decoder_reconstruction = classifier_model(classifier_model(img))
            # Plot original images
            plt.figure(figsize=(16, 16))
            images_per_row = 10
            num_rows = num_images // images_per_row
            for j in range(num_rows):
                for i in range(images_per_row):
                    plt.subplot(2 * num_rows, images_per_row, 2 * j * images_per_row + i + 1)
                    plt.imshow(ae_reconstruction[j * num_rows + i][0].cpu().numpy(), cmap='gray')
                    plt.title(f'AE')
                    plt.axis('off')
                    plt.subplot(2 * num_rows, images_per_row, (2 * j + 1) * images_per_row + i + 1)
                    plt.imshow(decoder_reconstruction[j * num_rows + i][0].cpu().numpy(), cmap='gray')
                    plt.title(f'C&D')
                    plt.axis('off')
            plt.show()
            break


def run_q3():
    classifier_based_encoder = networks.AutoEncoder(12)
    classifier_based_encoder.encoder.load_state_dict(torch.load(f'q2_encoder_model.pth'))
    classifier_based_encoder.encoder.eval()

    train_dataloader, test_dataloader = dl.get_dataloaders()
    # decoder = networks.Decoder(10)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(classifier_based_encoder.decoder.parameters(), lr=0.001)

    epochs = 30
    train_loss_list, test_loss_list = [], []
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loss = model_train(classifier_based_encoder, train_dataloader, optimizer, criterion)
        test_loss = model_test(classifier_based_encoder, test_dataloader, criterion)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        print(f'Train loss: {train_loss}, Test loss: {test_loss}')
    torch.save(classifier_based_encoder.state_dict(), f'q3_model.pth')
    plot_epochs_loss(train_loss_list, test_loss_list)


def run_trained_model():
    autoencoder = networks.AutoEncoder(12)
    autoencoder.load_state_dict(torch.load(f'q1_model.pth'))
    classifier_based_encoder = networks.AutoEncoder(12)
    classifier_based_encoder.load_state_dict(torch.load(f'q3_model.pth'))
    _, test_dataloader = dl.get_dataloaders()
    plot_results(classifier_based_encoder, autoencoder, test_dataloader)


if __name__ == "__main__":
    run_q3()
    run_trained_model()
