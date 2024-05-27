import matplotlib.pyplot as plt
import torch
import torchvision
from torchmetrics.classification import MulticlassConfusionMatrix
import numpy as np
import os
from torchvision.models import ResNet18_Weights
from torchvision.models import ResNet50_Weights
import torch.nn as nn

from sklearn.metrics import classification_report
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


from model import SmallCNN

def plotLossGraph(train_losses,valid_losses):
    # Plot the training and validation loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss Curves')
    plt.show()


def evaluation(test_loader, result_folder, best_model, model_type = 'resnet18', num_classes = 4, class_names = ['5.0','6.0','7.0','8.0']):
    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # Load the certain model
    if model_type == 'resnet18':
      model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
      model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_type == 'resnet50':
      model = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
      model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_type == 'smallcnn':
      model = SmallCNN(num_classes=num_classes)

    
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    # Load the weights from the best_model state_dict
    model.load_state_dict(torch.load(os.path.join(result_folder,best_model)))

    # Set the model in evaluation mode
    model.eval()

    # Initialize variables to store the true labels and predicted labels
    original_images = []
    true_labels = []
    predicted_labels = []
    file_names = []

    # Evaluate the model on the test dataset
    with torch.no_grad():
        for images, labels, file_name in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            # Append true and predicted labels to the respective lists
            original_images.extend(images.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())
            file_names.extend(file_name)
            

    # Convert lists to NumPy arrays
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    # Calculate the test accuracy
    test_accuracy = 100 * np.sum(true_labels == predicted_labels) / len(true_labels)
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
   # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(10, 10))
    # Set the global font size
    plt.rcParams.update({'font.size': 22})
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')
    # Set the font size of x and y axis labels
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.set_xlabel('Predicted pH', fontsize=20)
    ax.set_ylabel('True pH', fontsize=20)
    # Save the plot to a file
    file_path = os.path.join(result_folder, 'confusion_matrix.png')
    plt.savefig(file_path)
    plt.close(fig)

    # f1 score
    report = classification_report(true_labels, predicted_labels)

    # Print and return the classification report
    print("Classification Report:")
    print(report)

    results = {'original_images': original_images, 'true_labels': true_labels, 'predicted_labels': predicted_labels, 'file_names': file_names}
    return model, results