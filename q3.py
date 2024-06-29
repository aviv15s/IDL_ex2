import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from tqdm import tqdm
import networks
from utils import get_dataloaders, plot_graphs_train_test_values, model_test_images


def model_train(model, dataloader, optimizer, criterion):
    """
    custom training function that allows freezing an internal model (the encoder)
    :param model:
    :param dataloader:
    :param optimizer:
    :param criterion:
    :return:
    """
    model.train()
    model.encoder.eval()

    total_loss = 0
    for data in tqdm(dataloader):
        img, _ = data

        # Forward pass
        output = model(img)
        loss = criterion(output.to(torch.float64), img.to(torch.float64))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(img)

    return total_loss / len(dataloader.dataset)


def plot_results(classifier_model, ae_model, dataloader):
    """
    plots 50 images of digits reconstructed with both models
    :param classifier_model:
    :param ae_model:
    :param dataloader:
    :return:
    """
    num_images = 50
    classifier_model.eval()
    ae_model.eval()

    with torch.no_grad():
        counter = 1
        for data in dataloader.dataset:  # TODO select single batch
            counter += 1
            if counter > 50:
                break
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


def run_q3(epochs):
    classifier_based_encoder = networks.AutoEncoder(12)
    classifier_based_encoder.encoder.load_state_dict(torch.load(f'q2_encoder_model.pth'))

    train_dataloader, test_dataloader = get_dataloaders()
    criterion = nn.L1Loss()
    optimizer = optim.Adam(classifier_based_encoder.decoder.parameters(), lr=0.001)

    train_loss_list, test_loss_list = [], []
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loss = model_train(classifier_based_encoder, train_dataloader, optimizer, criterion)
        test_loss = model_test_images(classifier_based_encoder, test_dataloader, criterion)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        print(f'Train loss: {train_loss}, Test loss: {test_loss}')
    torch.save(classifier_based_encoder.state_dict(), f'q3_model.pth')
    plot_graphs_train_test_values(train_loss_list, test_loss_list, "loss", "loss [arb]", 3)


if __name__ == "__main__":
    run_q3(30)
