import numpy as np
import pandas as pd
from PIL import Image, ImageChops
import matplotlib.pyplot as plt
from skimage import color
from skimage.color import rgb2lab
import os


def concat_images(source, output_path, direction='horizontal', resize_dim=(512, 512)):
    '''Concatenate images horizontally or vertically'''
    files = sorted(os.listdir(source))
    image_paths = [os.path.join(source, img) for img in files]
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
    '''Get the median RGB and LAB values for the center crop of an image'''
    file_name = os.path.basename(image_path)
    with Image.open(image_path) as img:
        img = img.convert('RGB')
        img = img.resize(resize_dim)
        img_np = np.array(img)
        
        # Convert RGB to LAB
        img_lab = rgb2lab(img_np)

        # Calculate the coordinates for the central crop
        left = (resize_dim[0] - crop_dim[0]) // 2
        upper = (resize_dim[1] - crop_dim[1]) // 2
        right = left + crop_dim[0]
        lower = upper + crop_dim[1]

        img_cropped = img.crop((left, upper, right, lower))
        img_np_cropped = np.array(img_cropped)
        pixels_rgb = img_np_cropped.reshape(-1, 3)

        # Calculate the median for each RGB channel
        median_r = np.median(pixels_rgb[:, 0])
        median_g = np.median(pixels_rgb[:, 1])
        median_b = np.median(pixels_rgb[:, 2])

        img_lab_cropped = img_lab[upper:lower, left:right]
        pixels_lab = img_lab_cropped.reshape(-1, 3)

        # Calculate the median for each LAB channel
        median_L = round(np.median(pixels_lab[:, 0]),1)
        median_A = round(np.median(pixels_lab[:, 1]),1)
        median_B = round(np.median(pixels_lab[:, 2]),1)

        pH = file_name.split('_')[3]

        return file_name, median_r, median_g, median_b, median_L, median_A, median_B, pH

def generate_csv_from_folder(root_path, output_csv):
    '''Generate a CSV file with median RGB and LAB values for images in a folder using pandas'''
    data = []
    pH = ['50', '60', '70', '80']
    for ph in pH:
        folder_path = os.path.join(root_path, ph)
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                try:
                    row = get_median_rgb_center(file_path)
                    data.append(row)
                except Exception as e:
                    print(f"Error processing file {filename}: {e}")

    df = pd.DataFrame(data, columns=['file_name', 'r', 'g', 'b', 'L', 'A', 'B', 'pH'])
    df.to_csv(output_csv, index=False)

    print(f"CSV file '{output_csv}' created successfully.")

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


if __name__ == '__main__':
    # generate_csv_from_folder('./content/Dataset/Paper0725_divided', './content/Dataset/Paper0725_divided/median_rgb_values.csv')
    # root_path = './content/Dataset/PaperImages/Shuyi/new'
    # output = './content/Dataset/PaperImages/Shuyi_70.jpg'
    # concat_images(root_path, output, direction='horizontal', resize_dim=(400, 200))
    root_path = './content/Dataset/PaperImages/Yue'
    target_path = './content/Dataset/PaperImages/Yue_resized'
    pH = ['']
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    for ph in pH:
        path = os.path.join(root_path, ph)
        for filename in os.listdir(path):
            if not filename.endswith('.jpg'):
                continue
            file_path = os.path.join(path, filename)
            
            output_path = os.path.join(target_path, filename)
            resize_image(file_path, output_path, resize_dim=(450, 200))

