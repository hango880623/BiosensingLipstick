import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.transforms.functional import crop
from skimage.io import imread
from skimage.color import rgb2gray

from skimage.filters import threshold_otsu
from PIL import Image 

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
        image_np = image_rgb_tensor.permute(1, 2, 0).numpy()  # Convert to HWC format
        
        # Convert the NumPy array from RGB to LAB
        image_np_uint8 = (image_np * 255).astype(np.uint8)  # Convert to uint8
        image_lab = cv2.cvtColor(image_np_uint8, cv2.COLOR_RGB2LAB)
        
        # Convert the LAB image back to a tensor
        image_lab_tensor = torch.from_numpy(image_lab).permute(2, 0, 1).float() / 255.0
        
        # Make a new tuple that includes original and the path
        tuple_with_lab_and_path = (image_rgb_tensor, original_tuple[1], file_name)
        return tuple_with_lab_and_path

def mask_lips(img_path, threshold):
    # Load the image
    img = imread(img_path)
    img_gray = rgb2gray(img)
    # Apply the threshold to create a binary mask
    img_mask = img_gray < threshold
    # Apply the mask to the original image
    img_masked = img.copy()
    img_masked[~img_mask] = 0  # Set non-lips pixels to black
    return img_masked

def crop1000(img):  
    return crop(img, 0, 0, 1000, 1000)

def dataLoader_diff(base_path):
    # Set hyperparameters
    batch_size = 8

    # Initialize transformations for data augmentation
    transform = transforms.Compose([
        transforms.Resize((512,512)),
        transforms.CenterCrop(312),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # transforms.Lambda(crop1000),
        transforms.RandomCrop((224,224)),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2), # brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
        transforms.ToTensor(),
        # transforms.Normalize((0.58, 0.38, 0.5), (0.02, 0.02, 0.01))
    ])
    origin_dataset = ImageFolderWithPaths(
        root=base_path,
        transform=transform
    )
    origin_loader = torch.utils.data.DataLoader(origin_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    return origin_loader

def dataLoader_seperate(base_path):
    # Set hyperparameters
    batch_size = 8

    # Initialize transformations for data augmentation
    transform = transforms.Compose([
        transforms.Resize((512,512)),
        # transforms.CenterCrop(324),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # transforms.Lambda(crop1000),
        transforms.RandomCrop((224,224)),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2), # brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
        transforms.ToTensor(),
        # transforms.Normalize((0.58, 0.38, 0.5), (0.02, 0.02, 0.01))
    ])
    origin_dataset = ImageFolderWithPaths(
        root=base_path,
        transform=transform
    )
    origin_loader = torch.utils.data.DataLoader(origin_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    return origin_loader

def dataLoader(base_path):
    # Set hyperparameters
    batch_size = 8

    # Initialize transformations for data augmentation
    transform = transforms.Compose([
        transforms.Resize((512,512)),
        # transforms.CenterCrop(324),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # transforms.Lambda(crop1000),
        transforms.RandomCrop((224,224)),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2), # brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
        transforms.ToTensor(),
        # transforms.Normalize((0.58, 0.38, 0.5), (0.02, 0.02, 0.01))
    ])

    # Load the ImageNet Object Localization Challenge dataset
    origin_dataset = ImageFolderWithPaths(
        root=base_path,
        transform=transform
    )

    n = len(origin_dataset)  # total number of examples
    n_test = int(0.1 * n)  # take ~10% for test
    n_validation = int(0.2 * n)  # take ~10% for validation
    print('training: ',n-n_test-n_validation,'test: ',n_test,'validation: ',n_validation)

    train_dataset, validation_dataset , test_dataset= torch.utils.data.random_split(origin_dataset, [n-n_test-n_validation, n_test,n_validation])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    valid_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    return train_loader, valid_loader, test_loader


def api_dataLoader(base_path):

    # Set hyperparameters
    batch_size = 8

    # Initialize transformations for data augmentation
    transform = transforms.Compose([
        transforms.Resize((512,512)),
        transforms.CenterCrop(312),
        transforms.CenterCrop((224,224)),
        # transforms.Resize((128,128)),
        transforms.ToTensor(),
        # transforms.Normalize((0.58, 0.38, 0.5), (0.02, 0.02, 0.01))
    ])

    # Load the ImageNet Object Localization Challenge dataset
    origin_dataset = ImageFolderWithPaths(
        root=base_path,
        transform=transform
    )
    data_loader = torch.utils.data.DataLoader(origin_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    return data_loader

def random_dataLoader(base_path):

    # Set hyperparameters
    batch_size = 8

    # Initialize transformations for data augmentation
    transform = transforms.Compose([
        transforms.Resize((512,512)),
        transforms.CenterCrop((224,224)),
        # transforms.Resize((128,128)),
        transforms.ToTensor(),
        # transforms.Normalize((0.58, 0.38, 0.5), (0.02, 0.02, 0.01))
    ])

    # Load the ImageNet Object Localization Challenge dataset
    origin_dataset = ImageFolderWithPaths(
        root=base_path,
        transform=transform
    )
    data_loader = torch.utils.data.DataLoader(origin_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    return data_loader

def compute_mean_and_std_lab(dataloader):
    mean = np.zeros(3)
    std = np.zeros(3)
    nb_samples = 0

    for data in dataloader:
        images, _ , _= data
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
    train_loader, valid_loader, test_loader = dataLoader("./content/PixelCanon")

    # Compute the mean and standard deviation of the LAB color space
    mean, std = compute_mean_and_std_lab(train_loader)
    print(f'Mean: {mean}, Standard deviation: {std}')