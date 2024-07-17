from data_loader import dataLoaderLOPO, dataLoader_cross, dataloader_origin_test
from evaluations import evaluation
from training import train
from testing import test
import torch
import time
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import os


def lopo_analyze():
    '''Leave One Participant Out analysis'''
    folder_path = './content/Dataset/Paper_divided/'
    lopo = []
    for id in range(5):
        train_loader, valid_loader, test_loader = dataLoaderLOPO(
            folder_path, id)
        # model_types = ['customresnet','resnet50','resnet18', 'smallcnn']
        result_folder = './results/results_2024-07-07-dataset-paperlopodivid-resnet18/'

        start_time = time.time()
        model_path = train(result_folder, train_loader, valid_loader, model_type='resnet18',
                           num_epochs=51, learning_rate=0.001, pretrained=False, num_classes=4)

        end_time = time.time()

        elapsed_time = end_time - start_time
        print(f"Training time: {elapsed_time} seconds")

        elapsed_time_minutes = elapsed_time / 60

        print(f"Training time: {elapsed_time_minutes:.2f} minutes")

        # test
        model, test_plots, test_acc = evaluation(
            test_loader, result_folder, 'best_50.pth', model_type='resnet18', num_classes=4)
        lopo.append(test_acc)
    print(lopo)


def k_fold_cross_validation():
    '''K-Fold Cross Validation analysis'''
    base_path = './content/Dataset/Paper_center_crop_divided/'
    k = 5
    origin_dataset, test_dataset = dataloader_origin_test(base_path)
    kf = KFold(n_splits=k, shuffle=True)
    fold = 0
    all_fold_accuracies = []
    result_folder = './results/results_2024-07-16-dataset-papercrossvalid-resnet18/'

    for train_idx, val_idx in kf.split(origin_dataset):
        print(f'Fold {fold + 1}/{k}')
        train_loader, valid_loader = dataLoader_cross(
            origin_dataset, train_idx, val_idx)
        folder_name = result_folder + f'results_fold_{fold + 1}/'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        best_model_name = train(folder_name, train_loader, valid_loader, model_type='resnet18',
                                num_epochs=101, learning_rate=0.001, pretrained=False, num_classes=4)
        all_fold_accuracies.append(best_model_name)
        fold += 1

    print('Cross-validation results:')
    for i, model_name in enumerate(all_fold_accuracies):
        print(f'Fold {i + 1}: Best model saved as {model_name}')

    test(base_path, result_folder)


if __name__ == "__main__":
    torch.manual_seed(99)
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        x = torch.ones(1, device=mps_device)
        print(x)
    else:
        print("MPS device not found.")
    k_fold_cross_validation()
