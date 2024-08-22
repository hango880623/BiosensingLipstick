import torch
from torch.utils.data import DataLoader, Subset
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import numpy as np
import cv2
import os


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths and converts images to LAB color space. Extends
    torchvision.datasets.ImageFolder.
    """

    def __getitem__(self, index):
        # This is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # The image file path
        path = self.imgs[index][0]
        file_name = os.path.basename(path)

        # Convert the image tensor to a NumPy array
        image_rgb_tensor = original_tuple[0]
        image_np = image_rgb_tensor.permute(
            1, 2, 0).numpy()  # Convert to HWC format

        # Convert the NumPy array from RGB to LAB
        image_np_uint8 = (image_np * 255).astype(np.uint8)  # Convert to uint8
        image_lab = cv2.cvtColor(image_np_uint8, cv2.COLOR_RGB2LAB)

        # Convert the LAB image back to a tensor
        image_lab_tensor = torch.from_numpy(
            image_lab).permute(2, 0, 1).float() / 255.0

        # Make a new tuple that includes original and the path
        tuple_with_lab_and_path = (
            image_rgb_tensor, original_tuple[1], file_name)
        return tuple_with_lab_and_path

# transform = transforms.Compose([
#         transforms.Resize((512, 512)),
#         transforms.CenterCrop(324),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomVerticalFlip(),
#         transforms.RandomCrop((224, 224)),
#         transforms.ToTensor(),
#     ])

def dataloader_origin_test(base_path):
    '''Load the dataset and split it into training and test sets'''

    # # Initialize transformations for data augmentation
    # transform = transforms.Compose([
    #     transforms.Resize((512, 512)),
    #     transforms.CenterCrop(324),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomVerticalFlip(),
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    # ])

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        # transforms.CenterCrop(324),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # transforms.RandomCrop((224, 224)),
        transforms.ToTensor(),
    ])

    # Load the ImageNet Object Localization Challenge dataset
    origin_dataset = ImageFolderWithPaths(
        root=base_path,
        transform=transform
    )

    n = len(origin_dataset)  # total number of examples
    n_test = int(0.1 * n)  # take ~10% for test
    print('training: ', n-n_test, 'test: ', n_test)

    train_dataset, test_dataset = torch.utils.data.random_split(
        origin_dataset, [n-n_test, n_test])

    return train_dataset, test_dataset


def dataLoader_cross(origin_dataset, train_idx, val_idx):
    '''Create data loaders for training and validation sets'''
    batch_size = 8

    train_dataset = Subset(origin_dataset, train_idx)
    validation_dataset = Subset(origin_dataset, val_idx)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    valid_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    return train_loader, valid_loader


def dataLoader(base_path):
    '''Load the dataset and split it into training, validation, and test sets, including data augmentation'''
    batch_size = 8

    # Initialize transformations for data augmentation
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        # transforms.CenterCrop(324),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # transforms.Lambda(crop1000),
        transforms.RandomCrop((224, 224)),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2), # brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
        transforms.ToTensor(),
        # transforms.Normalize((0.58, 0.38, 0.5), (0.02, 0.02, 0.01))
    ])

    origin_dataset = ImageFolderWithPaths(
        root=base_path,
        transform=transform
    )

    n = len(origin_dataset)  # total number of examples
    n_test = int(0.1 * n)  # take ~10% for test
    n_validation = int(0.2 * n)  # take ~10% for validation
    print('training: ', n-n_test-n_validation, 'test: ',
          n_test, 'validation: ', n_validation)

    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(
        origin_dataset, [n-n_test-n_validation, n_test, n_validation])

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    valid_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    return train_loader, valid_loader, test_loader


def api_dataLoader(base_path):
    '''Load the dataset for the API'''
    batch_size = 8

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.CenterCrop(312),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
    ])

    origin_dataset = ImageFolderWithPaths(
        root=base_path,
        transform=transform
    )
    data_loader = torch.utils.data.DataLoader(
        origin_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    return data_loader


def dataLoaderLOPO(base_path, leave_out_id):
    '''Load the dataset and split it into training, validation, and test sets. For Leave One Participant Out purposes'''
    batch_size = 8

    # transform = transforms.Compose([
    #     transforms.Resize((512, 512)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomVerticalFlip(),
    #     transforms.RandomCrop((224, 224)),
    #     transforms.ToTensor(),
    # ])

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        # transforms.CenterCrop(324),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # transforms.RandomCrop((224, 224)),
        transforms.ToTensor(),
    ])

    origin_dataset = ImageFolderWithPaths(
        root=base_path,
        transform=transform
    )

    # Split dataset into train, validation, and test sets based on participant ID
    train_data = []
    test_data = []

    for idx in range(len(origin_dataset)):
        image, label, file_name = origin_dataset[idx]
        participant_id = int(file_name.split('_')[2])
        if participant_id == leave_out_id:
            test_data.append((image, label, file_name))
        else:
            train_data.append((image, label, file_name))

    # Further split the training data into training and validation sets
    train_size = int(0.8 * len(train_data))
    validation_size = len(train_data) - train_size
    train_dataset, validation_dataset = torch.utils.data.random_split(
        train_data, [train_size, validation_size])

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    valid_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=True, num_workers=1)

    return train_loader, valid_loader, test_loader


def compute_mean_and_std_lab(dataloader):
    mean = np.zeros(3)
    std = np.zeros(3)
    nb_samples = 0

    for data in dataloader:
        images, _, _ = data
        batch_samples = images.size(0)

        for img in images:
            img_np = img.permute(1, 2, 0).numpy()
            img_np_uint8 = (img_np * 255).astype(np.uint8)
            img_lab = cv2.cvtColor(img_np_uint8, cv2.COLOR_RGB2LAB)
            mean += img_lab.mean(axis=(0, 1))
            std += img_lab.std(axis=(0, 1))

        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    return mean / 255.0, std / 255.0


if __name__ == "__main__":
   # Example usage:
    train_loader, valid_loader, test_loader = dataLoader(
        "./content/PixelCanon")

    # Compute the mean and standard deviation of the LAB color space
    mean, std = compute_mean_and_std_lab(train_loader)
    print(f'Mean: {mean}, Standard deviation: {std}')
