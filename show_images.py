import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import torchvision.utils as vutils
import numpy as np
from PIL import Image


def showDataset(base_path):
    '''Define a function to display all images in the certain folder'''
    data_path = base_path + '55/'
    image_filenames = os.listdir(data_path)

    # Choose the first 25 image filenames
    image_filenames = image_filenames[:25]

    # Calculate the number of rows and columns for the subplot grid
    num_rows = 5
    num_cols = math.ceil(len(image_filenames) / num_rows)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))

    # Loop through the images and display them in subplots
    for i, ax in enumerate(axes.flat):
        if i < len(image_filenames):
            img_path = os.path.join(data_path, image_filenames[i])
            img = mpimg.imread(img_path)
            ax.imshow(img)
            ax.set_title(image_filenames[i])
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def show_all_images(data_loader):
    '''Define a function to display all images in the test dataset'''
    all_images = []

    for images, _ in data_loader:
        all_images.extend(images)

    # Create a grid of images
    image_grid = vutils.make_grid(all_images, nrow=4, padding=5)
    image_grid = image_grid.cpu().numpy().transpose((1, 2, 0))

    plt.figure(figsize=(12, 6))
    plt.imshow(image_grid)
    plt.axis('off')
    plt.show()


def show_evaluation_images(results, base_path, result_folder, class_labels=None, num_cols=4, show=False):
    '''Define a function to display evaluation images'''
    original_images = results['original_images']
    true_labels = results['true_labels']
    predicted_labels = results['predicted_labels']
    file_names = results['file_names']
    num_images = len(original_images)
    num_rows = (num_images + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 3 * num_rows))

    for i, filename in enumerate(file_names):
        row = i // num_cols
        col = i % num_cols
        true_label = true_labels[i]
        predicted_label = predicted_labels[i]
        ax = axes[row, col]
        path = os.path.join(base_path, class_labels[true_label])
        image = Image.open(os.path.join(path, filename))
        ax.imshow(image)
        ax.set_title(
            f'{file_names[i]}\nTrue: {class_labels[true_label]} Predicted: {class_labels[predicted_label]}')
        ax.axis('off')

    # Remove empty subplots, if any
    for i in range(len(original_images), num_rows * num_cols):
        fig.delaxes(axes.flatten()[i])

    plt.tight_layout()
    if show:
        file_path = os.path.join(result_folder, 'testimage.png')
        plt.savefig(file_path)
        plt.show()

    plt.close(fig)
