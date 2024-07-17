import torch
from torch.utils.data import DataLoader
from data_loader import dataloader_origin_test
from evaluations import evaluation


def test(base_path,result_folder):
    origin_dataset, test_dataset = dataloader_origin_test(base_path)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True, num_workers=1)
    model, test_plots, test_acc = evaluation(test_loader, result_folder,'best_50.pth', model_type = 'resnet18', num_classes = 4)

if __name__ == "__main__":
    torch.manual_seed(99)
    base_path = './content/Dataset/Paper_center_crop_divided/'
    result_folder = './results/results_2024-07-16-dataset-papercrossvalid-resnet18/'
    test(base_path,result_folder)
    