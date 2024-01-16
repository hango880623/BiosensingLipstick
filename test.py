import matplotlib.pyplot as plt
import torch

# Define a function to display all images in the test loader with true and predicted labels as subplots
def show_all_images_with_labels(test_loader, model, class_labels=None, num_cols=4):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    num_images = len(test_loader.dataset)
    num_rows = (num_images + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 3 * num_rows))

    with torch.no_grad():
        image_idx = 0
        for images, true_labels in test_loader:
            images = images.to(device)
            true_labels = true_labels.to(device)
            outputs = model(images)
            _, predicted_labels = torch.max(outputs, 1)

            for j in range(images.size(0)):
                if image_idx >= num_images or image_idx > 20:
                    break

                image = images[j].cpu().numpy().transpose((1, 2, 0))
                true_label = true_labels[j].cpu().item()
                predicted_label = predicted_labels[j].cpu().item()

                # Get the class label names (if available)
                true_label_name = class_labels[true_label] if class_labels is not None else true_label
                predicted_label_name = class_labels[predicted_label] if class_labels is not None else predicted_label

                # Calculate the row and column indices for the subplot
                row = image_idx // num_cols
                col = image_idx % num_cols

                # Display the image with true and predicted labels as the title
                ax = axes[row, col]
                ax.imshow(image)
                ax.set_title(f'True: {true_label_name}\nPredicted: {predicted_label_name}')
                ax.axis('off')

                image_idx += 1

                if image_idx >= num_images:
                    break

    # Remove empty subplots, if any
    for i in range(image_idx, num_rows * num_cols):
        fig.delaxes(axes.flatten()[i])

    plt.tight_layout()
    plt.show()