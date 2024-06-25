from matplotlib import pyplot as plt
from torch import nn, optim

import data_loaders as dl
import networks
import torch

import q2


def decoder_train(decoder_model, classifier_model, dataloader, optimizer, criterion):
    best_loss = 10000
    decoder_model.train()
    classifier_model.eval()

    epoch_num = 10
    for epoch in range(epoch_num):
        for data in dataloader:
            img, label = data

            # Forward pass
            output = decoder_model(classifier_model(img))
            loss = criterion(output, img)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            best_loss = min(best_loss, loss.item())
        print(f'Epoch [{epoch + 1}/{epoch_num}], Loss: {loss.item():.4f}')
    return best_loss


def plot_results(classifier_model, ae_model, decoder_model, dataloader):
    num_images = 50

    classifier_model.eval()
    ae_model.eval()
    decoder_model.eval()

    with torch.no_grad():
        for data in dataloader:  # TODO select single batch
            img, label = data
            ae_reconstruction = ae_model(img)
            decoder_reconstruction = decoder_model(classifier_model(img))
            # Plot original images
            plt.figure(figsize=(16, 16))
            images_per_row = 10
            num_rows = num_images // images_per_row
            for j in range(num_rows):
                for i in range(images_per_row):
                    plt.subplot(2*num_rows, images_per_row, 2*j*images_per_row + i + 1)
                    plt.imshow(ae_reconstruction[j*num_rows + i][0].cpu().numpy(), cmap='gray')
                    plt.title(f'AE')
                    plt.axis('off')

                    plt.subplot(2 * num_rows, images_per_row, (2 * j + 1) * images_per_row + i + 1)
                    plt.imshow(decoder_reconstruction[j * num_rows + i][0].cpu().numpy(), cmap='gray')
                    plt.title(f'C&D')
                    plt.axis('off')

            plt.show()
            break

def run_q3():

    classifier_model_path = input("Enter Classifier Model Path:\n")
    classifier = networks.Classifier(10)
    classifier.load_state_dict(torch.load(classifier_model_path))
    classifier.eval()

    autoencoder_model_path = input("Enter AutoEncoder Model Path:\n")
    autoencoder = networks.AutoEncoder(12)
    autoencoder.load_state_dict(torch.load(autoencoder_model_path))
    autoencoder.eval()

    train_dataloader, test_dataloader = dl.get_dataloaders()
    decoder = networks.Decoder(10)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(decoder.parameters(), lr=0.001)
    train_loss = decoder_train(decoder, classifier, train_dataloader, optimizer, criterion)

    torch.save(decoder.state_dict(), f'q3_model_{round(train_loss, 2)}.pth')

def run_trained_model():
    autoencoder_model_path = r"C:\Users\t9092556\PycharmProjects\IDL_ex2\q1_model_0.05.pth"
    classifier_model_path = r"C:\Users\t9092556\PycharmProjects\IDL_ex2\q2_model_0.0.pth"
    decoder_model_path = r"C:\Users\t9092556\PycharmProjects\IDL_ex2\q3_model_0.07.pth"

    # autoencoder_model_path = input("Enter AutoEncoder Model Path:\n")
    autoencoder = networks.AutoEncoder(12)
    autoencoder.load_state_dict(torch.load(autoencoder_model_path))
    autoencoder.eval()

    # classifier_model_path = input("Enter Classifier Model Path:\n")
    classifier = networks.Classifier(10)
    classifier.load_state_dict(torch.load(classifier_model_path))
    classifier.eval()

    # decoder_model_path = input("Enter Decoder Model Path:\n")
    decoder = networks.Decoder(10)
    decoder.load_state_dict(torch.load(decoder_model_path))
    decoder.eval()

    _, test_dataloader = dl.get_dataloaders()
    plot_results(classifier, autoencoder, decoder, test_dataloader)

if __name__ == "__main__":
    run_trained_model()
    # run_q3()