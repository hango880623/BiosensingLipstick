import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms
import numpy as np
from torchvision.models import ResNet18_Weights
from torchvision.models import ResNet50_Weights
import torch.nn as nn
from PIL import Image
import io

import datetime

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((512,512)),
        transforms.CenterCrop(312),
        transforms.CenterCrop((224,224)),
        transforms.ToTensor(),
])
pH_value = {0: '5.0', 1: '6.0', 2: '7.0',3: '8.0'}

# Load the certain model
def load_model(model_path, model_type = 'resnet18'):
    num_classes = 4
    if model_type == 'resnet18':
      model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
      model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_type == 'resnet50':
      model = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
      model.fc = nn.Linear(model.fc.in_features, num_classes)

    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    # Load the weights from the best_model state_dict
    model.load_state_dict(torch.load(model_path))
    return model

def predict(model,image):
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model.eval()
    # Convert binary data to PIL Image
    image_pil = Image.open(io.BytesIO(image))
    image_pil = image_pil.rotate(90, expand=True)
    image_pil.save("./user/"+ current_datetime+ '_image_full.jpg')
    image_pil = image_pil.crop((500, 2000, 1800, 2700)) #(700, 750, 1200, 1750)
    image_pil.save("./user/"+ current_datetime+ '_image_crop.jpg')

    # Preprocess the image
    image_tensor = transform(image_pil).unsqueeze(0)

    # Save the image tensor as an image file
    
    

    # Evaluate the model on the preprocessed image
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs.data, 1)
    
    return predicted
