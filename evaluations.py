import matplotlib.pyplot as plt
import torch
import torchvision
from torchmetrics.classification import MulticlassConfusionMatrix
import numpy as np


from model import CustomResNetModel

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

def evaluation(test_loader, best_model, model_type = 'resnet18'):
    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # Load the certain model
    if model_type == 'resnet18':
      model = torchvision.models.resnet18(weights=True)
    elif model_type == 'resnet50':
      model = torchvision.models.resnet50(weights=True)
    elif model_type == 'customresnet':
      model = CustomResNetModel()

    model = torch.nn.DataParallel(model)
    model = model.to(device)
    # Load the weights from the best_model state_dict
    model.load_state_dict(torch.load(best_model))

    # Set the model in evaluation mode
    model.eval()

    # Initialize variables to store the true labels and predicted labels
    original_images = []
    true_labels = []
    predicted_labels = []

    # Evaluate the model on the test dataset
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            # Append true and predicted labels to the respective lists
            original_images.extend(images.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

    # Convert lists to NumPy arrays
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    if model_type != 'customresnet':
        # Calculate the test accuracy
        test_accuracy = 100 * np.sum(true_labels == predicted_labels) / len(true_labels)
        print(f"Test Accuracy: {test_accuracy:.2f}%")
        metric = MulticlassConfusionMatrix(num_classes=5)
        metric.update(torch.tensor(predicted_labels),torch.tensor(true_labels))
        fig_, ax_ = metric.plot()
    results = {'original_images': original_images, 'true_labels': true_labels, 'predicted_labels': predicted_labels}
    return model, results
