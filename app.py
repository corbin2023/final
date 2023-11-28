import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

# Load the PyTorch model
model = torch.load('best_rgmodel.pth', map_location=torch.device('cpu'))
model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to make predictions
def predict(image):
    img = Image.open(image).convert('RGB')
    img = transform(img)
    img = img.unsqueeze(0)  # Add batch dimension
    output = model(img)
    return output

# Title
st.title('Corbin App - Ruber Guardian')

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the selected image
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Make prediction
    prediction = predict(uploaded_file)

    # Apply softmax to convert logits to probabilities
    probabilities = F.softmax(prediction, dim=1)

    # Extract the probability for each class
    prob_good = probabilities[0, 1].item()
    prob_defect = probabilities[0, 0].item()

    # Define a threshold (you may need to adjust this based on your model)
    threshold = 0.5

    # Interpret the prediction
    result = "good" if prob_good >= threshold else "defect"

    st.write("Probability of being good:", prob_good)
    st.write("Probability of being defect:", prob_defect)
    st.write("Prediction:", result)
