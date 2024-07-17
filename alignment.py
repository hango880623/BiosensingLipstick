import numpy as np
import pandas as pd
from PIL import Image, ImageChops
import matplotlib.pyplot as plt
from skimage import color
from skimage.color import rgb2lab


def concat_images(image_paths, output_path, direction='horizontal', resize_dim=(512, 512)):
    '''Concatenate images horizontally or vertically'''
    images = [Image.open(img_path) for img_path in image_paths]

    if direction == 'horizontal':
        total_width = sum(img.width for img in images)
        max_height = max(img.height for img in images)
        concatenated_image = Image.new('RGB', (total_width, max_height))

        x_offset = 0
        for img in images:
            concatenated_image.paste(img, (x_offset, 0))
            x_offset += img.width
    elif direction == 'vertical':
        total_height = sum(img.height for img in images)
        max_width = max(img.width for img in images)
        concatenated_image = Image.new('RGB', (max_width, total_height))

        y_offset = 0
        for img in images:
            concatenated_image.paste(img, (0, y_offset))
            y_offset += img.height
    else:
        raise ValueError("Direction must be 'horizontal' or 'vertical'")

    # Save the concatenated image
    concatenated_image.save(output_path)


def ChopDifference():
    '''Find the difference between the base image and the target images'''
    base = Image.open('./image/p_2_00_6500_pixel.jpg')

    # ['50', '55', '60', '65', '70', '75', '80']
    pH = ['55', '60', '65', '70', '80']

    # Create a figure
    plt.figure(figsize=(15, len(pH) * 5))

    for i, ph in enumerate(pH):
        path = './image/p_2_' + ph + '_6500_pixel.jpg'
        target = Image.open(path)

        # Finding difference
        diff = ImageChops.difference(target, base)

        print(base.size, target.size, diff.size)
        # Convert images to NumPy arrays
        image_base_np = np.array(base)
        image_target_np = np.array(target)
        diff_np = np.array(diff)

        # Display images using Matplotlib
        plt.subplot(len(pH), 3, 1 + i * 3)
        plt.title('Original Image')
        plt.imshow(image_base_np)
        plt.axis('off')

        plt.subplot(len(pH), 3, 2 + i * 3)
        plt.title('Target Image ' + ph)
        plt.imshow(image_target_np)
        plt.axis('off')

        plt.subplot(len(pH), 3, 3 + i * 3)
        plt.title('Difference Image')
        plt.imshow(diff_np)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('./image/plot2.png')


def rgb_to_lab(image):
    """Convert an RGB image to LAB color space."""
    image_np = np.array(image)
    lab_image = color.rgb2lab(image_np)
    return Image.fromarray(lab_image.astype(np.uint8))


def get_median_rgb_center(image_path, resize_dim=(512, 512), crop_dim=(224, 224)):
    '''Get the median RGB values for the center crop of an image'''
    with Image.open(image_path) as img:
        img = img.convert('RGB')
        img = img.resize(resize_dim)
        # Convert RGB to LAB
        img_lab = rgb2lab(img_np)

        # Calculate the coordinates for the central crop
        left = (resize_dim[0] - crop_dim[0]) // 2
        upper = (resize_dim[1] - crop_dim[1]) // 2
        right = left + crop_dim[0]
        lower = upper + crop_dim[1]

        img_cropped = img.crop((left, upper, right, lower))
        img_np = np.array(img_cropped)
        pixels = img_np.reshape(-1, 3)

        # Calculate the median for each channel
        median_r = np.median(pixels[:, 0])
        median_g = np.median(pixels[:, 1])
        median_b = np.median(pixels[:, 2])

        return median_r, median_g, median_b


def save_to_csv(x_values, y_values, output_path):
    data = np.hstack((x_values, y_values.reshape(-1, 1)))
    df = pd.DataFrame(data, columns=['r', 'g', 'b', 'y'])
    df.to_csv(output_path, index=False)
    print(f'Data saved to {output_path}')


def resize_image(image_path, output_path, resize_dim=(512, 512)):
    with Image.open(image_path) as img:
        img_resized = img.resize(resize_dim)
        img_resized.save(output_path)
        print(f"Resized image saved to {output_path}")
