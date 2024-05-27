from load_data import cleanDS_Store
from data_loader import dataLoader, dataLoader_seperate, dataLoader_diff
from training import train
import torch
import time

if __name__ == "__main__":
    torch.manual_seed(99)
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        x = torch.ones(1, device=mps_device)
        print (x)
    else:
        print ("MPS device not found.")

    # train_path = './content/LOPOPixelDiff/TrainDiff/'
    # valid_path = './content/LOPOPixelDiff/ValidationDiff/'
    # test_path = './content/LOPOPixelDiff/TestDiff/'
    # train_loader, valid_loader, test_loader = dataLoader_diff(train_path), dataLoader_diff(valid_path), dataLoader_diff(test_path)

    folder_path = './content/Dataset/PaperLOPOChange/Train'
    train_loader, valid_loader, test_loader = dataLoader(folder_path)
    # model_types = ['customresnet','resnet50','resnet18', 'smallcnn']
    result_folder = './results/results_2024-05-24-dataset-paperlopochange-resnet18/'

    start_time = time.time()
    model_path = train(result_folder, train_loader, valid_loader, model_type = 'resnet18', num_epochs = 101, learning_rate = 0.001, pretrained=False, num_classes = 4)

    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Training time: {elapsed_time} seconds")

    elapsed_time_minutes = elapsed_time / 60

    print(f"Training time: {elapsed_time_minutes:.2f} minutes")