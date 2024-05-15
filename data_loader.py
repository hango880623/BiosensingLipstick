import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.functional import crop

def crop800(image):
    return crop(image, 52, 82, 128, 128)

def crop1000(image):
    return crop(image, 45, 120, 192, 216)

def dataLoader(base_path):
    # Set hyperparameters
    batch_size = 8

    # Initialize transformations for data augmentation
    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.RandomHorizontalFlip(),
        # transforms.Lambda(crop1000),
        # transforms.RandomCrop(216),
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
        transforms.Resize((256,256)),
        # transforms.CenterCrop(216),
        transforms.ToTensor()
    ])

    # Load the ImageNet Object Localization Challenge dataset
    origin_dataset = torchvision.datasets.ImageFolder(
        root=base_path,
        transform=transform
    )
    data_loader = torch.utils.data.DataLoader(origin_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    return data_loader