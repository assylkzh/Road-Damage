import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from model import EnhancedResNet50  # Replace with your model import

# Load your trained model
@st.cache_resource
def load_model():
    model = EnhancedResNet50(num_classes=4)  # Adjust based on your final model
    model.load_state_dict(torch.load("", map_location=torch.device('cpu')))
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
