import torch
import torchvision
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models import ResNet18_Weights
from torchvision.models import ResNet50_Weights


from model import CustomResNetModel
import numpy as np

import os
import datetime

def train(train_loader, valid_loader, model_type = 'resnet18', num_epochs = 50, learning_rate = 0.001):
    # Set device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print('Now using: ',device)
    print(model_type)
    # Load the certain model
    if model_type == 'resnet18':
      model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
      # Define the loss function and optimizer
      criterion = torch.nn.CrossEntropyLoss()
      optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
      print("model: resnet18")
    elif model_type == 'resnet50':
      model = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
      criterion = torch.nn.CrossEntropyLoss()
      optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
      print("model: resnet50")
    elif model_type == 'customresnet':
      model = CustomResNetModel()
      criterion = torch.nn.MSELoss()
      optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
      print("model: customresnet")

    model = torch.nn.DataParallel(model)
    model = model.to(device)

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5) #, verbose=True
    # print("Current learning rate: ", scheduler.get_last_lr())

    # Define variables to keep track of the best accuracy and corresponding model
    best_accuracy = 0.0
    best_loss = 1000
    best_model = None

    # Define empty lists to store training and validation losses
    train_losses = []
    valid_losses = []

    # Get the current date and time
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Create a folder with the current date and time
    folder_name = os.path.join('results',f"results_{current_datetime}")
    os.makedirs(folder_name)

    # Train the model...
    for epoch in range(num_epochs):
        running_loss = 0.0

        for inputs, labels in train_loader:
            # Zero out the optimizer
            optimizer.zero_grad()
            # Move input and label tensors to the device
            inputs = inputs.to(device)
            # Forward pass
            outputs = model(inputs)
            if model_type == 'customresnet':
                labels = labels.view(-1, 1).float().to(device)
            else:
                labels = labels.to(device)

            # print(outputs, labels)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Calculate the average training loss for this epoch
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}')

        # Step the learning rate scheduler
        scheduler.step(avg_train_loss)
        if model_type == 'resnet18' or  model_type == 'resnet50':
            # Validation part
            with torch.no_grad():
                valid_loss = 0.0
                correct = 0
                total = 0

                for images, labels in valid_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    valid_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                # Calculate average validation loss and accuracy
                avg_valid_loss = valid_loss / len(valid_loader)
                valid_losses.append(avg_valid_loss)
                accuracy = 100 * correct / total

                print(f'Accuracy of the network on the validation images: {accuracy:.2f}%')
                # print(f'Validation Loss: {avg_valid_loss:.4f}')

                # Save the model if it has the highest accuracy so far
                if accuracy >= best_accuracy:
                    best_accuracy = accuracy
                    best_model = model.state_dict()  # Save model weights
        else:
            with torch.no_grad():
                valid_loss = 0.0
                correct = 0
                total = 0

                for images, labels in valid_loader:
                    images = images.to(device)
                    labels = labels.view(-1, 1).float().to(device)

                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    valid_loss += loss.item()
                # print(outputs,labels)
                # Calculate average validation loss and accuracy
                avg_valid_loss = valid_loss / len(valid_loader)
                valid_losses.append(avg_valid_loss)

                print(f'Validation Loss: {avg_valid_loss:.4f}')

                # Save the model if it has the highest accuracy so far
                if best_loss > avg_valid_loss:
                    best_loss = avg_valid_loss
                    best_model = model.state_dict()  # Save model weights


    # Save the best model to a file
    save_name = f'best_{model_type}_{learning_rate}_{num_epochs}.pth'
    save_path = os.path.join(folder_name, save_name)
    torch.save(best_model, save_path)
    if model_type == 'resnet18' or  model_type == 'resnet50':
        print(f'Finished Training, Best Validation Accuracy: {best_accuracy:.2f}%')
    else:
        print(f'Finished Training, Best Validation Loss: {best_loss:.2f}')


    # Convert train_losses and valid_losses to numpy arrays
    train_losses_arr = np.array(train_losses)
    valid_losses_arr = np.array(valid_losses)

    # Save train_losses and valid_losses to text files
    np.savetxt(os.path.join(folder_name, 'train_losses.txt'), train_losses_arr)
    np.savetxt(os.path.join(folder_name, 'valid_losses.txt'), valid_losses_arr)

    return train_losses, valid_losses, save_name

