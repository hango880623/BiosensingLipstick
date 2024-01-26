import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import torchvision.utils as vutils
import numpy as np

def showDataset(base_path):
    data_path = base_path + '55/'

    # Get a list of image filenames in the folder
    image_filenames = os.listdir(data_path)

    # Choose the first 25 image filenames
    image_filenames = image_filenames[:25]

    # Calculate the number of rows and columns for the subplot grid
    num_rows = 5  # Change this to control the number of rows
    num_cols = math.ceil(len(image_filenames) / num_rows)

    # Create a subplot grid
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

# Define a function to display all images in the test dataset
def show_all_images(data_loader):
    all_images = []

    for images, _ in data_loader:
        all_images.extend(images)

    # Create a grid of images
    image_grid = vutils.make_grid(all_images, nrow=4, padding=5)

    # Convert the tensor to a NumPy array
    image_grid = image_grid.cpu().numpy().transpose((1, 2, 0))

    # Display the grid of images
    plt.figure(figsize=(12, 6))
    plt.imshow(image_grid)
    plt.axis('off')
    plt.show()

def show_evaluation_images(results,base_path, class_labels=None, num_cols=4):
    original_images = results['original_images']
    true_labels = results['true_labels']
    predicted_labels = results['predicted_labels']
    num_images = len(original_images)
    num_rows = (num_images + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 3 * num_rows))

    for i,image in enumerate(original_images):
    # Calculate the row and column indices for the subplot
        row = i // num_cols
        col = i % num_cols
        true_label = true_labels[i]
        predicted_label = predicted_labels[i]
        # Display the image with true and predicted labels as the title
        ax = axes[row, col]
        image = np.transpose(image, (1, 2, 0))
        ax.imshow(image)
        ax.set_title(f'True: {class_labels[true_label]}\nPredicted: {class_labels[predicted_label]}')
        ax.axis('off')


    # Remove empty subplots, if any
    for i in range(len(original_images), num_rows * num_cols):
        fig.delaxes(axes.flatten()[i])

    plt.tight_layout()
    plt.show()