import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms
import numpy as np

from PIL import Image
import io

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize(256),
    # transforms.CenterCrop(216),
    transforms.ToTensor()
])

# Load the certain model
def load_model(model_path, model_type = 'resnet18'):
    
    if model_type == 'resnet18':
      model = torchvision.models.resnet18(weights=True)
    elif model_type == 'resnet50':
      model = torchvision.models.resnet50(weights=True)

    model = torch.nn.DataParallel(model)
    model = model.to(device)
    # Load the weights from the best_model state_dict
    model.load_state_dict(torch.load(model_path))
    return model

def predict(model,image):
    model.eval()
    # Convert binary data to PIL Image
    image_pil = Image.open(io.BytesIO(image))
    # Preprocess the image
    image_tensor = transform(image_pil).unsqueeze(0)

    # Evaluate the model on the preprocessed image
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs.data, 1)
    
    return predicted
