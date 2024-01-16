import os
import shutil


def testSkinTones():
    # Define the source and target directories
    source_directories = [
        "/content/Canon/lips",
        "/content/Pixel/lips"
    ]
    target_directory = "/content/SkinTone/lips"

    # List of subdirectories to combine
    subdirectories_to_combine = ['55','60','65','70','75','80']

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
                        if file == '.DS_Store':
                            continue
                        file_des = file.split('_')
                        if file_des[-3] != '1': # remove p1
                            source_file = os.path.join(root, file)
                            target_file = os.path.join(subdirectory_path, file)
                            shutil.copy(source_file, target_file)

def testDevices():
    # Define the source and target directories
    source_directories = [
        "/content/Canon/lips",
        "/content/Pixel/lips"
    ]
    target_directory = "/content/Device/lips"

    # List of subdirectories to combine
    subdirectories_to_combine = ['55','60','65','70','75','80']

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
                        if source_dir == "/content/Canon/lips":
                            source_file = os.path.join(root, file)
                            target_file = os.path.join(subdirectory_path, file)
                            shutil.copy(source_file, target_file)

def testLightConditions():
    # Define the source and target directories
    source_directories = [
        "/content/Canon/lips",
        "/content/Pixel/lips"
    ]
    target_directory = "/content/LightCondition/lips"

    # List of subdirectories to combine
    subdirectories_to_combine = ['55','60','65','70','75','80']

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
                        if file == '.DS_Store':
                            continue
                        file_des = file.split('_')
                        if file_des[-1] != '5000.jpg' or file_des[-1] != '2200.jpg': # remove 5000K light condition
                            source_file = os.path.join(root, file)
                            target_file = os.path.join(subdirectory_path, file)
                            shutil.copy(source_file, target_file)

def testPHvalues():
    # Define the source and target directories
    source_directories = [
        "/content/Canon/lips",
        "/content/Pixel/lips"
    ]
    target_directory = "/content/pH/lips"

    # List of subdirectories to combine
    subdirectories_to_combine = ['55','60','65','70','75','80'] # ,'75' neglect pH=7.5

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
                        if subdirectory != '80':
                            source_file = os.path.join(root, file)
                            target_file = os.path.join(subdirectory_path, file)
                            shutil.copy(source_file, target_file)