import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.transforms.functional import crop
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from PIL import Image 

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder.
    """
    def __getitem__(self, index):
        # This is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # The image file path
        path = self.imgs[index][0]
        # Make a new tuple that includes original and the path
        tuple_with_path = (original_tuple[0], original_tuple[1], path)
        return tuple_with_path

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

def crop800(image):
    return crop(image, 52, 82, 128, 128)

def crop1000(image):
    return crop(image, 45, 120, 192, 216)

def dataLoader(base_path):
    # Set hyperparameters
    batch_size = 8

    # Initialize transformations for data augmentation
    transform = transforms.Compose([
        transforms.Resize((512,512)),
        transforms.CenterCrop(256),
        transforms.RandomHorizontalFlip(),
        # transforms.Lambda(crop1000),
        transforms.RandomCrop((128,128)),
        transforms.ToTensor()
    ])

    # Load the ImageNet Object Localization Challenge dataset
    origin_dataset = torchvision.datasets.ImageFolder(
        root=base_path,
        transform=transform
    )

    n = len(origin_dataset)  # total number of examples
    n_test = int(0.1 * n)  # take ~10% for test
    n_validation = int(0.1 * n)  # take ~10% for validation
    print('training: ',n-n_test-n_validation,'test: ',n_test,'validation: ',n_validation)

    train_dataset, validation_dataset , test_dataset= torch.utils.data.random_split(origin_dataset, [n-n_test-n_validation, n_test,n_validation])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    valid_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    return train_loader, valid_loader, test_loader

def original_dataLoader(base_path):

    # Set hyperparameters
    batch_size = 8

    # Initialize transformations for data augmentation
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(216),
        transforms.ToTensor()
    ])

    # Load the ImageNet Object Localization Challenge dataset
    origin_dataset = torchvision.datasets.ImageFolder(
        root=base_path,
        transform=transform
    )

    n = len(origin_dataset)  # total number of examples
    n_test = int(n)  # take ~10% for test
    n_validation = int(0)  # take ~10% for validation
    print('training: ',n-n_test-n_validation,'test: ',n_test,'validation: ',n_validation)

    train_dataset, validation_dataset , test_dataset= torch.utils.data.random_split(origin_dataset, [n-n_test-n_validation, n_test,n_validation])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    valid_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    return train_loader, valid_loader, test_loader

def ExpDataLoader(base_path):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set hyperparameters
    batch_size = 8

    # Initialize transformations for data augmentation
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    # Load the ImageNet Object Localization Challenge dataset
    origin_dataset = torchvision.datasets.ImageFolder(
        root=base_path,
        transform=transform
    )

    # Create a DataLoader for the entire dataset
    data_loader = torch.utils.data.DataLoader(origin_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    return data_loader

def api_dataLoader(base_path):

    # Set hyperparameters
    batch_size = 8

    # Initialize transformations for data augmentation
    transform = transforms.Compose([
        transforms.Resize((512,512)),
        transforms.CenterCrop(256),
        transforms.CenterCrop((128,128)),
        # transforms.Resize((128,128)),
        transforms.ToTensor()
    ])

    # Load the ImageNet Object Localization Challenge dataset
    origin_dataset = torchvision.datasets.ImageFolder(
        root=base_path,
        transform=transform
    )
    data_loader = torch.utils.data.DataLoader(origin_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    return data_loader

if __name__ == "__main__":
    # Read image 
    image = Image.open('/Users/kuyuanhao/Documents/Imyphone/iMyFone D-Back for Mac/D-Back Recovery/PC_Recover/D-Back for Mac 20240517-064555/All Files/Photo/JPG/6P8A4491.JPG') 
    
    # create an transform for crop the image 
    transform = transforms.Compose([
        transforms.Resize((512,512)),
        transforms.CenterCrop(320),
        transforms.RandomResizedCrop(size=(128, 128),scale=(0.2, 0.8))
    ])
    # use above created transform to crop 
    # the image 
    image_crop = transform(image) 
    
    # display result 
    image_crop.show() 