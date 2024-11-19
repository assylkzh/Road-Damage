import torch
import torch.nn as nn
import torchvision.models as models
import streamlit as st
from PIL import Image
import torchvision.transforms as transforms

# Define the model class (same as in your notebook)
class EnhancedResNet50(nn.Module):
    def __init__(self, num_classes):
        super(EnhancedResNet50, self).__init__()
        self.base_model = models.resnet50(pretrained=True)
        
        # Freeze the base model layers if required
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Modify the final layers
        self.base_model.fc = nn.Sequential(
            nn.Linear(self.base_model.fc.in_features, 512),  # Reduce dimensionality
            nn.ReLU(),
            nn.BatchNorm1d(512),  # Batch Normalization
            nn.Dropout(0.4),      # Dropout with 40% probability
            nn.Linear(512, num_classes)  # Output layer
        )

    def forward(self, x):
        return self.base_model(x)

# Load the trained model
def load_model():
    model = EnhancedResNet50(num_classes=4)  # Adjust num_classes if needed
    state_dict = torch.load('road_damage_model2.pth', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict, strict=False)  # Set strict=False to ignore missing keys
    model.eval()  # Set model to evaluation mode
    return model

model = load_model()

# Define the class names
class_names = ["Good", "Poor", "Satisfactory", "Very Poor"]

# Define image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to model's expected input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Streamlit app
st.title("Road Damage Classification")
st.write("Upload an image of a road to classify its condition.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")  # Open and convert image to RGB
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Processing...")

    # Preprocess and predict
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted_class = torch.max(outputs, 1)

    st.write(f"Predicted Road Condition: **{class_names[predicted_class.item()]}**")
