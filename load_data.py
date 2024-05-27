import os
import shutil
import zipfile

def unzip_file(zip_file_path, extract_to_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_path)
    # Remove the __MACOSX folder
    macosx_path = os.path.join(extract_to_path, '__MACOSX')
    if os.path.exists(macosx_path):
        shutil.rmtree(macosx_path)

def cleanDS_Store(base_path):
    image_foldernames = os.listdir(base_path)
    for file in image_foldernames:
      if file == '.DS_Store':
          os.remove(os.path.join(base_path,'.DS_Store'))

    classes = os.listdir(base_path)
    for c in classes:
      data_path = base_path + c
      image_filenames = os.listdir(data_path)
      if len(image_filenames) == 0:
        os.rmdir(data_path)
        continue
      for file in image_filenames:
        if file == '.DS_Store':
            os.remove(os.path.join(data_path,'.DS_Store'))

def loadData():
    # Make the directory if it doesn't exist
    directories = [
        "./content/Canon",
        "./content/Pixel",
        "./content/CanonF",
        "./content/PixelF",
        "./content/IphoneF",
        "./content/Canon0119",
        "./content/Canon0122"
    ]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
    # Define the source and target directories
    # source_directories = [
    #     "./content/Canon/lips",
    #     "./content/Pixel/lips",
    #     "./content/CanonF/lips",
    #     "./content/PixelF/lips",
    #     "./content/IphoneF/lips"
    # ]
    # unzip_file('./content/zipfile/lips1004_Canon.zip', './content/Canon/')
    # unzip_file('./content/zipfile/lips1004_Pixel.zip', './content/Pixel/')
    # unzip_file('./content/zipfile/lips1130_Canon.zip', './content/CanonF/')
    # unzip_file('./content/zipfile/lips1130_Pixel.zip', './content/PixelF/')
    # unzip_file('./content/zipfile/lips1130_Iphone.zip', './content/IphoneF/')
    # source_directories = ["./content/Canon0119/lips","./content/Pixel0119/lips" ]
    # unzip_file('./content/zipfile/lips0116_Canon.zip', './content/Canon0119/')
    # unzip_file('./content/zipfile/lips0116_Pixel.zip', './content/Pixel0119/')
    source_directories = ["./content/Canon0122/lips"] # ,"./content/Canon0119/lips","./content/Pixel0119/lips"

    target_directory = "./content/Fake/lips"

    # List of subdirectories to combine
    subdirectories_to_combine = ['55', '60', '65', '70', '80']

    # Create the target directory if it doesn't exist
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    # Combine the subdirectories
    for subdirectory in subdirectories_to_combine:
        subdirectory_path = os.path.join(target_directory, subdirectory)
        if not os.path.exists(subdirectory_path):
            os.makedirs(subdirectory_path)

        for source_dir in source_directories:
            source_subdirectory = os.path.join(source_dir, subdirectory)
            if os.path.exists(source_subdirectory):
                for root, _, files in os.walk(source_subdirectory):
                    for file in files:
                        source_file = os.path.join(root, file)
                        unique_identifier = source_dir.split('/')[-2]
                        filename, extension = os.path.splitext(file)
                        new_filename = f"{filename}_{unique_identifier}{extension}"

                        target_file = os.path.join(subdirectory_path, new_filename)
                        shutil.copy(source_file, target_file)

if __name__ == "__main__":
    loadData()