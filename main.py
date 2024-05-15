from load_data import cleanDS_Store
from data_loader import dataLoader
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
    base_exp_path = ['./content/SkinTone/lips/','./content/Device/lips/','./content/LightCondition/lips/','./content/pH/lips/']
    
    full_path = './content/GenAI/lips/'
    base_path = full_path
    cleanDS_Store(base_path)
    train_loader, valid_loader, test_loader = dataLoader(base_path)

    start_time = time.time()
    # model_types = ['customresnet','resnet50','resnet18']
    train_losses, valid_losses, model_path = train(train_loader, valid_loader, model_type = 'resnet18', num_epochs = 50, learning_rate = 0.001)

    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Training time: {elapsed_time} seconds")

    elapsed_time_minutes = elapsed_time / 60

    print(f"Training time: {elapsed_time_minutes:.2f} minutes")