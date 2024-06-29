import numpy as np
import torch

import utils as dl
import matplotlib.pyplot as plt


def plot_digits_histogram():
    train_dataloader, test_dataloader = dl.get_dataloaders()
    train_dataset, test_dataset = train_dataloader.dataset, test_dataloader.dataset
    train_histogram, test_histogram = {i: 0 for i in range(10)}, {i: 0 for i in range(10)}

    for _, label in train_dataset:
        train_histogram[label] += 1

    for _, label in test_dataset:
        test_histogram[label] += 1

    # Create a figure and axis
    fig, ax = plt.subplots()
    plt.title("Digits Histogram")
    plt.ylabel("#Count")

    # Define bar width. We'll use this to offset the second bar.
    bar_width = 0.35

    # Positions of the left bar-boundaries
    bar_l = np.arange(len(train_histogram))

    # Positions of the x-axis ticks (center of the bars as bar labels)
    tick_pos = [i + (bar_width / 2) for i in bar_l]

    # Create the total score bar
    ax.bar(bar_l, train_histogram.values(), width=bar_width, color='b', alpha=0.5, label='Train Dataset')

    # Create the final score bar
    ax.bar(bar_l + bar_width, test_histogram.values(), width=bar_width, color='r', alpha=0.5, label='Test Dataset')

    # Set the x ticks with names
    plt.xticks(tick_pos, test_histogram.keys())

    # Add legend and show the plot
    ax.legend()
    plt.show()



def plot_average_digit():
    train_dataloader, test_dataloader = dl.get_dataloaders()
    train_dataset, test_dataset = train_dataloader.dataset, test_dataloader.dataset
    # train_average, test_average = {i: torch.mean(train_dataset[]) for i in range(10)}, {i: 0 for i in range(10)}

    # indices = [(idx for idx, target in enumerate(train_dataset.targets) if target == i) for i in range(10)]
    # subset = {i: torch.mean(torch.stack(list(torch.utils.data.Subset(train_dataset, indices[i]))), dim=0) for i in range(10)}
    #
    # # tensors = [dataset[i] for i in range(len(dataset))]
    # # stacked_tensors = torch.stack(tensors)
    # # average_tensor = torch.mean(stacked_tensors, dim=0)
    #

    train_average, test_average = {label: torch.zeros([28, 28]) for label in range(10)}, {label: torch.zeros([28, 28]) for label in range(10)}
    train_count, test_count = {label: 0 for label in range(10)}, {label: 0 for label in range(10)}

    # Iterate over the dataset
    for i in range(len(train_dataset)):
        image, label = train_dataset[i]
        train_average[label] += image[0]
        train_count[label] += 1

    # Iterate over the dataset
    for i in range(len(test_dataset)):
        image, label = test_dataset[i]
        test_average[label] += image[0]
        test_count[label] += 1

    train_average = {i: train_average[i] / train_count[i] for i in range(10)}
    test_average = {i: test_average[i] / test_count[i] for i in range(10)}

    plt.figure(figsize=(12, 4))
    plt.suptitle("Average digit of each type in test and train datasets")

    for i in range(10):
        plt.subplot(2, 10, i+1)
        plt.imshow(train_average[i].cpu().numpy(), cmap='gray')
        plt.title(f'train {i}')
        plt.axis('off')

        plt.subplot(2, 10, i + 1 + 10)
        plt.imshow(test_average[i].cpu().numpy(), cmap='gray')
        plt.title(f'test {i}')
        plt.axis('off')
    plt.show()


if __name__ == "__main__":
    # plot_digits_histogram()
    plot_average_digit()
