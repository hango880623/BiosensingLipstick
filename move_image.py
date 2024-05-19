import os
from PIL import Image

dir = ['55','60','65','70','80']
def separate_image(folder,without_folder, with_folder):
    for pH in dir:
        path = os.path.join(folder,pH)
        for filename in os.listdir(path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                input_file_path = os.path.join(path, filename)
                if filename.split('_')[1] == '0':
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
    """
    Center-crops an image to the specified output size.
    
    Args:
        image_path (str): Path to the input image file.
        output_size (tuple): Output size in the format (width, height).
    
    Returns:
        PIL.Image: Center-cropped image.
    """
    image = Image.open(image_path)
    width, height = image.size
    target_width, target_height = output_size
    
    left = (width - target_width) / 2
    top = (height - target_height) / 2
    right = (width + target_width) / 2
    bottom = (height + target_height) / 2
    
    cropped_image = image.crop((left, top, right, bottom))
    return cropped_image

def center_crop_images_in_folder(folder_path, output_size, output_folder):
    """
    Center-crops all images in a folder to the specified output size.
    
    Args:
        folder_path (str): Path to the folder containing images.
        output_size (tuple): Output size in the format (width, height).
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            input_image_path = os.path.join(folder_path, filename)
            output_image_path = os.path.join(output_folder, f"cropped_{filename}")
            
            cropped_image = center_crop_image(input_image_path, output_size)
            cropped_image.save(output_image_path)
            print(f"Center-cropped {filename} saved as cropped_{filename}")

def divide_image_into_four(image_path, output_folder):
    """
    Divides an image into four equal parts and saves them to the output folder.
    
    Args:
        image_path (str): Path to the input image file.
        output_folder (str): Path to the folder where the output images will be saved.
    """
    image = Image.open(image_path)
    width, height = image.size
    
    # Calculate dimensions for each quadrant
    half_width = width // 2
    half_height = height // 2
    
    # Define the box coordinates for each quadrant
    # boxes = [
    #     (0, 0, half_width, half_height),         # Top-left
    #     (half_width, 0, width, half_height),     # Top-right
    #     (0, half_height, half_width, height),    # Bottom-left
    #     (half_width, half_height, width, height) # Bottom-right
    # ]
    boxes = [(0,0,width,height),(0,0,width,height),(0,0,width,height),(0,0,width,height)]
    
    # Crop and save each quadrant
    base_filename = os.path.basename(image_path)
    filename_without_ext, ext = os.path.splitext(base_filename)
    
    for i, box in enumerate(boxes):
        quadrant = image.crop(box)
        output_image_path = os.path.join(output_folder, f"{filename_without_ext}_part{i+1}{ext}")
        quadrant.save(output_image_path)
        print(f"Saved {output_image_path}")

def divide_images_in_folder(folder_path, output_folder):
    """
    Divides all images in a folder into four equal parts and saves them to the output folder.
    
    Args:
        folder_path (str): Path to the folder containing images.
        output_folder (str): Path to the folder where the output images will be saved.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            input_image_path = os.path.join(folder_path, filename)
            divide_image_into_four(input_image_path, output_folder)

def label_0516(directory):
    lights = ["6500", "5700", "5000", "4000", "3500", "3000", "2700", "2200"]
    ph_values = ["00", "50", "55", "60", "65", "70", "75", "80"]
    settings = ["custom","daylight","Shade"]
    # Iterate through each image file in the directory
    names = sorted(os.listdir(directory))
    names = [item for item in names if not item.startswith('.')]
    print(names)
    for index, filename in enumerate(names):
        if filename.endswith(".JPG"):
            # Extract the light and ph values based on the index
            light = lights[index % len(lights)]
            ph = ph_values[index // len(lights) // len(settings) % len(ph_values)]
            setting = settings[index // len(lights)  % len(settings)]
            # Construct the new filename
            new_filename = f"p_4_{ph}_{light}_{setting}.JPG"

            # Full paths to the old and new files
            old_filepath = os.path.join(directory, filename)
            new_filepath = os.path.join(directory, new_filename)

            # Rename the file
            os.rename(old_filepath, new_filepath)

    print("Files have been renamed successfully.")

if __name__ == "__main__":
    # # separate image example
    # folder = '/Users/kuyuanhao/Documents/Research Assistant/Interactive Organisms Lab/data/Pixel/0116/lipsnew'
    # without_folder = '/Users/kuyuanhao/Documents/Research Assistant/Interactive Organisms Lab/data/Pixel/0116/lips_without'
    # with_folder = '/Users/kuyuanhao/Documents/Research Assistant/Interactive Organisms Lab/data/Pixel/0116/lips_with'
    # separate_image(folder,without_folder, with_folder)

    # # center crop image example
    # folder_path = '/Users/kuyuanhao/Documents/Research Assistant/Interactive Organisms Lab/data/Canon/0116/lipscrop'
    # output_folder_path = '/Users/kuyuanhao/Documents/Research Assistant/Interactive Organisms Lab/data/Canon/0116/lipscropped_center'
    # for pH in dir:
    #     target_path = os.path.join(folder_path, pH)
    #     output_size = (750, 300)
    #     center_crop_images_in_folder(target_path, output_size,output_folder_path)

    # # example for label_0516
    # folder_path = '/Users/kuyuanhao/Documents/Research Assistant/Interactive Organisms Lab/data/Canon/0516new'
    # label_0516(folder_path)

    # # example for separate_image
    input_folder_path = "/Users/kuyuanhao/Documents/Research Assistant/Interactive Organisms Lab/data/Canon/0116/lips"
    output_folder_path = "/Users/kuyuanhao/Documents/Research Assistant/Interactive Organisms Lab/data/Canon/0116/lips_cropped"
    for pH in dir:
        input_folder = os.path.join(input_folder_path, pH)
        output_folder = os.path.join(output_folder_path, pH)
        divide_images_in_folder(input_folder,output_folder)