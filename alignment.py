from PIL import Image, ImageChops 
import matplotlib.pyplot as plt
import numpy as np
import os
from skimage import color

import pandas as pd

from sklearn.linear_model import LinearRegression
from load_data import cleanDS_Store

from skimage.color import rgb2lab


def concat_images(image_paths, output_path, direction='horizontal', resize_dim=(512, 512)):
    # images = [Image.open(img_path).resize(resize_dim) for img_path in image_paths]
    images = [Image.open(img_path) for img_path in image_paths]

    # Determine the width and height of the concatenated image
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

def pil_to_np(image):
    return np.array(image)

def ChopDifference():
    # Example usage
    base = Image.open('./image/p_2_00_6500_pixel.jpg')
    
    pH = ['55', '60', '65', '70', '80'] #['50', '55', '60', '65', '70', '75', '80']


    # Create a figure
    plt.figure(figsize=(15, len(pH) * 5))

    for i, ph in enumerate(pH):
        path = './image/p_2_' + ph + '_6500_pixel.jpg'
        target = Image.open(path)

        # Finding difference
        diff = ImageChops.difference(target, base)

        print(base.size, target.size, diff.size)
        # Convert images to NumPy arrays
        image_base_np = pil_to_np(base)
        image_target_np = pil_to_np(target)
        diff_np = pil_to_np(diff)
    
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
    # plt.show()

def rgb_to_lab(image):
    """Convert an RGB image to LAB color space."""
    image_np = np.array(image)
    lab_image = color.rgb2lab(image_np)
    return Image.fromarray(lab_image.astype(np.uint8))

def chop(base, target):
    # Assuming chop is a function that computes the difference between two images
    return ImageChops.difference(base, target)

def main():
    src_folder = '/Users/kuyuanhao/Documents/Research Assistant/Interactive Organisms Lab/data/Dataset/Paper/00'
    trg_folder = '/Users/kuyuanhao/Documents/Research Assistant/Interactive Organisms Lab/data/Dataset/Paper'
    dst_folder = '/Users/kuyuanhao/Documents/Research Assistant/Interactive Organisms Lab/data/Dataset/PaperLOPOLAB'
    pH = ['50', '60', '70', '80']
    resize_dim = (512, 512)

    # Create the destination directory if it doesn't exist
    os.makedirs(dst_folder, exist_ok=True)

    for ph in pH:
        ph_folder = os.path.join(trg_folder, ph)
        dst_ph_folder = os.path.join(dst_folder, ph)
        os.makedirs(dst_ph_folder, exist_ok=True)
        
        files = os.listdir(ph_folder)
        for file in files:
            file_base = file[:4] + '00' + file[6:]
            base_path = os.path.join(src_folder, file_base)
            target_path = os.path.join(ph_folder, file)
            
            # Check if base image exists
            if os.path.exists(base_path):
                base = Image.open(base_path).resize(resize_dim)
                target = Image.open(target_path).resize(resize_dim)
                base_lab = rgb_to_lab(base)
                target_lab = rgb_to_lab(target)
                
                diff = chop(base_lab, target_lab)
                diff.save(os.path.join(dst_ph_folder, 'diff_' + file))
            else:
                print(f"Base image {file_base} not found.")


def get_rgb_sum(image_path):
    # Open the image using Pillow
    image = Image.open(image_path)

    # Convert the image to a NumPy array
    image_np = np.array(image)

    # Ensure the image has 3 color channels (RGB)
    if len(image_np.shape) == 3 and image_np.shape[2] == 3:
        # Sum the RGB values
        rgb_sum = image_np.sum(axis=(0, 1))
    else:
        raise ValueError("Image does not have 3 color channels (RGB)")

    return rgb_sum

def checkSum():
    trg_folder = './content/LOPOPixelDiff/TrainDiff/lips'
    pH = ['55', '60', '65', '70', '80']
    
    x_values = []
    y_values = []

    for ph in pH:
        ph_folder = os.path.join(trg_folder, ph)
        
        files = os.listdir(ph_folder)
        for file in files:
            target_path = os.path.join(ph_folder, file)
            value = get_rgb_sum(target_path)
            x_values.append(value)
            y = file.split('_')[3]
            y_values.append(y)
    
    return np.array(x_values), np.array(y_values)

def get_median_rgb_center(image_path, resize_dim=(512, 512), crop_dim=(224, 224)):
    # Open the image
    with Image.open(image_path) as img:
        # Convert the image to RGB mode if it's not already
        img = img.convert('RGB')
        # Resize the image to 512x512
        img = img.resize(resize_dim)
        
        # Calculate the coordinates for the central crop
        left = (resize_dim[0] - crop_dim[0]) // 2
        upper = (resize_dim[1] - crop_dim[1]) // 2
        right = left + crop_dim[0]
        lower = upper + crop_dim[1]
        
        # Crop the central 224x224 region
        img_cropped = img.crop((left, upper, right, lower))
        
        # Convert the cropped image to a NumPy array
        img_np = np.array(img_cropped)
        
        # Reshape the array to be a list of RGB values
        pixels = img_np.reshape(-1, 3)
        
        # Calculate the median for each channel
        median_r = np.median(pixels[:, 0])
        median_g = np.median(pixels[:, 1])
        median_b = np.median(pixels[:, 2])
        
        return median_r, median_g, median_b

def get_median_lab_center(image_path, resize_dim=(512, 512), crop_dim=(224, 224)):
    """Get the median LAB value from the central 224x224 region of a 512x512 image."""
    with Image.open(image_path) as img:
        img = img.convert('RGB')
        img = img.resize(resize_dim)
        
        left = (resize_dim[0] - crop_dim[0]) // 2
        upper = (resize_dim[1] - crop_dim[1]) // 2
        right = left + crop_dim[0]
        lower = upper + crop_dim[1]
        
        img_cropped = img.crop((left, upper, right, lower))
        img_np = np.array(img_cropped)
        
        # Convert RGB to LAB
        img_lab = rgb2lab(img_np)
        pixels = img_lab.reshape(-1, 3)
        
        median_l = np.median(pixels[:, 0])
        median_a = np.median(pixels[:, 1])
        median_b = np.median(pixels[:, 2])
        
        return median_l, median_a, median_b

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
    # main()
    # # image_paths = ['./image/p_0_00_6500_pixel.jpg', './image/p_0_55_6500_pixel.jpg', './image/p_0_60_6500_pixel.jpg', './image/p_0_65_6500_pixel.jpg', './image/p_0_70_6500_pixel.jpg','./image/p_0_80_6500_pixel.jpg']
    # image_paths = ['./image/p0.jpg','./image/p1.jpg','./image/p2.jpg','./image/p3.jpg','./image/p4.jpg']
    # output_path = './image/merge.jpg'
    # concat_images(image_paths, output_path, direction='horizontal', resize_dim=(512, 512))


    dir = ['00','10','50','60','70','80']
    base_path = '/Users/kuyuanhao/Documents/Research Assistant/Interactive Organisms Lab/data/Dataset/Paper resize'
    data = []
    for pH in dir:
        pH_path = os.path.join(base_path,pH)
        files = os.listdir(pH_path)
        for file in files:
            if file.lower().endswith('.jpg'):
                resize_image(os.path.join(pH_path,file),os.path.join(pH_path,file),resize_dim=(550, 250))
                # L,A,B = get_median_rgb_center(os.path.join(pH_path,file))
                # data.append({'file_name': file,'L': f'{L:.2f}', 'A': f'{A:.2f}', 'B': f'{B:.2f}', 'pH': pH})

    # # Create a DataFrame and write to CSV
    # df = pd.DataFrame(data)
    # output_path = os.path.join(base_path,'median_rgb_values.csv')
    # df.to_csv(output_path, index=False)
    # print(f'Data saved to {output_path}')