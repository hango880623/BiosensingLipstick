import os
import shutil
from PIL import Image
import pandas as pd
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
import random

dir = ['55', '60', '65', '70', '80']


def separate_image(folder, without_folder, with_folder, selected='0'):
    for pH in dir:
        path = os.path.join(folder, pH)
        for filename in os.listdir(path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                input_file_path = os.path.join(path, filename)
                if filename.split('_')[1] == selected:
                    output_folder = os.path.join(with_folder, pH)
                    if not os.path.exists(output_folder):
                        os.makedirs(output_folder)
                    output_file_path = os.path.join(output_folder, filename)
                else:
                    output_folder = os.path.join(without_folder, pH)
                    if not os.path.exists(output_folder):
                        os.makedirs(output_folder)
                    output_file_path = os.path.join(output_folder, filename)
                os.rename(input_file_path, output_file_path)


def center_crop_image(image_path, output_size):
    image = Image.open(image_path)
    width, height = image.size
    target_width, target_height = output_size

    left = (width - target_width) / 2
    top = (height - target_height) / 2
    right = (width + target_width) / 2
    bottom = (height + target_height) / 2

    cropped_image = image.crop((left, top, right, bottom))
    return cropped_image


def center_crop_images_in_folder(folder_path, output_folder, output_size):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            input_image_path = os.path.join(folder_path, filename)
            output_image_path = os.path.join(
                output_folder, f"cropped_{filename}")

            cropped_image = center_crop_image(input_image_path, output_size)
            cropped_image.save(output_image_path)
            print(f"Center-cropped {filename} saved as cropped_{filename}")


def divide_image(image_path, output_folder, rows=2, cols=4):
    img = Image.open(image_path)
    origin_filename = os.path.basename(image_path).split('.')[0]
    img_width, img_height = img.size

    part_width = img_width // cols
    part_height = img_height // rows

    for row in range(rows):
        for col in range(cols):
            if col == 0 or col == 3:
                continue
            left = col * part_width
            upper = row * part_height
            right = (col + 1) * part_width
            lower = (row + 1) * part_height

            part = img.crop((left, upper, right, lower))

            part_path = os.path.join(
                output_folder, f"{origin_filename}_{row}_{col}.jpg")
            part.save(part_path)
            print(f"Saved part {row},{col} to {part_path}")


def divide_images_in_folder(folder_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            input_image_path = os.path.join(folder_path, filename)
            divide_image(input_image_path, output_folder)


def label_0719(id, directory, device):
    lights = ["6500", "5700", "5000", "4000", "3500", "3000", "2700", "2200"]
    ph_values = ["00", "50", "60", "70", "80", "no"]
    types = ["s", "g"]
    names = sorted(os.listdir(directory))
    names = [item for item in names if not item.startswith('.')]
    print(names)
    for index, filename in enumerate(names):
        if filename.endswith((".jpg", ".png", ".jpeg",".JPG", ".PNG", ".JPEG", ".HEIC")):

            light = lights[index % len(lights)]
            ph = ph_values[index // len(lights) % len(ph_values)]
            type = types[index // len(lights) // len(ph_values)]
            new_filename = f"{type}_p_{id}_{ph}_{light}_{device}.jpg"

            old_filepath = os.path.join(directory, filename)
            new_filepath = os.path.join(directory, new_filename)
            os.rename(old_filepath, new_filepath)

    print("Files have been renamed successfully.")


def label_0516(directory):
    lights = ["6500", "5700", "5000", "4000", "3500", "3000", "2700", "2200"]
    ph_values = ["00", "50", "55", "60", "65", "70", "75", "80"]
    settings = ["custom", "daylight", "Shade"]
    # Iterate through each image file in the directory
    names = sorted(os.listdir(directory))
    names = [item for item in names if not item.startswith('.')]
    print(names)
    for index, filename in enumerate(names):
        if filename.endswith(".JPG"):
            light = lights[index % len(lights)]
            ph = ph_values[index // len(lights) //
                           len(settings) % len(ph_values)]
            setting = settings[index // len(lights) % len(settings)]
            new_filename = f"p_4_{ph}_{light}_{setting}.JPG"

            old_filepath = os.path.join(directory, filename)
            new_filepath = os.path.join(directory, new_filename)
            os.rename(old_filepath, new_filepath)

    print("Files have been renamed successfully.")


def get_pixel_values(image_path, points):
    image = Image.open(image_path).convert('RGB')
    pixel_values = []

    for point in points:
        pixel_value = image.getpixel(point)
        pixel_values.append(pixel_value)

    return pixel_values


def convert_rgb_to_lab(rgb):

    rgb_color = sRGBColor(rgb[0], rgb[1], rgb[2], is_upscaled=True)
    lab_color = convert_color(rgb_color, LabColor)
    return (lab_color.lab_l, lab_color.lab_a, lab_color.lab_b)


def save_pixel_values_to_csv(folders, points, output_csv):
    if os.path.exists(output_csv):
        df = pd.read_csv(output_csv)
    else:
        df = pd.DataFrame(columns=['Filename', 'X', 'Y', 'L', 'A', 'B'])

    rows = []
    for folder_path in folders:
        for filename in os.listdir(folder_path):
            if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                image_path = os.path.join(folder_path, filename)
                pixel_values = get_pixel_values(image_path, points)
                for point, pixel_value in zip(points, pixel_values):
                    L, A, B = convert_rgb_to_lab(pixel_value)
                    rows.append(
                        {'Filename': filename, 'X': point[0], 'Y': point[1], 'L': f'{L:.2f}', 'A': f'{A:.2f}', 'B': f'{B:.2f}'})

    new_df = pd.DataFrame(rows)
    # Concatenate the existing DataFrame with the new DataFrame
    if not new_df.empty:
        df = pd.concat([df, new_df], ignore_index=True)

    df.to_csv(output_csv, index=False)
    print(f"Saved pixel values to {output_csv}")


def resize_images(folder_path, resize_dim=(512, 512)):
    files = os.listdir(folder_path)

    for file_name in files:
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            file_path = os.path.join(folder_path, file_name)

            with Image.open(file_path) as img:
                resized_img = img.resize(resize_dim)
                resized_img.save(file_path)
                print(f'Resized: {file_path}')

points = [(86, 90), (128, 96), (170, 90), (50, 140), (192, 140), (72, 180), (128, 196), (186, 180)]

def separate_images_by_filename_segment(source_folder):
    if not os.path.exists(source_folder):
        print(f"Source folder '{source_folder}' does not exist.")
        return

    for filename in os.listdir(source_folder):
        segments = filename.split('_')
        if len(segments) > 3:
            segment = segments[3]
            destination_folder = os.path.join(source_folder, segment)
            if not os.path.exists(destination_folder):
                os.makedirs(destination_folder)
            
            source_path = os.path.join(source_folder, filename)
            destination_path = os.path.join(destination_folder, filename)
            shutil.move(source_path, destination_path)
            print(f"Moved '{filename}' to '{destination_folder}'")
        else:
            print(f"Filename '{filename}' does not have enough segments.")

def split_dataset(root_path, train_path, test_path, test_ratio=0.1):
    labels = ['50', '60', '70', '80']
    for label in labels:
        folder_path = os.path.join(root_path, label)
        images = os.listdir(folder_path)
        random.shuffle(images)
        
        split_point = int(len(images) * test_ratio)
        test_images = images[:split_point]
        train_images = images[split_point:]

        train_label_path = os.path.join(train_path, label)
        test_label_path = os.path.join(test_path, label)

        os.makedirs(train_label_path, exist_ok=True)
        os.makedirs(test_label_path, exist_ok=True)

        for image in train_images:
            shutil.copy(os.path.join(folder_path, image), os.path.join(train_label_path, image))

        for image in test_images:
            shutil.copy(os.path.join(folder_path, image), os.path.join(test_label_path, image))

        print(f"Label {label}: {len(train_images)} images in training, {len(test_images)} images in testing.")


if __name__ == "__main__":
    # Divide images example
    
    folder = ['50', '60', '70', '80']
    for pH in folder:
        # center_crop_images_in_folder(os.path.join('./content/Dataset/Paper',pH),os.path.join('./content/Dataset/Paper_center_crop',pH),(324,324))
        # resize_images(os.path.join('./content/Dataset/Paper_divided_256',pH),(512,512))
        divide_images_in_folder(os.path.join('./content/Dataset/Paper0725', pH),
                                os.path.join('./content/Dataset/Paper0725_divided', pH))
    

    # separate_images_by_filename_segment('/Users/kuyuanhao/Documents/Research Assistant/Interactive Organisms Lab/data/Paper0725')

    # # Example usage:
    # root_path = '/Users/kuyuanhao/Documents/Research Assistant/Interactive Organisms Lab/data/Paper0725'
    # train_path = '/Users/kuyuanhao/Documents/Research Assistant/Interactive Organisms Lab/data/Paper0725Train'
    # test_path = '/Users/kuyuanhao/Documents/Research Assistant/Interactive Organisms Lab/data/Paper0725Test'

    # split_dataset(root_path, train_path, test_path)