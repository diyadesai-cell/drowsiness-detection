import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import io

# ------------------------------------
# 1️⃣ Model Definition
# ------------------------------------
class SmallCNN(nn.Module):
    def __init__(self):
        super(SmallCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

# ------------------------------------
# 2️⃣ Load Model
# ------------------------------------
model_path = r"C:\Users\91973\Desktop\drowsiness dataset2\vs code training\smallcnn_drowsiness(4).pth"

device = torch.device("cpu")
model = SmallCNN()
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
else:
    st.error(f"❌ Model not found at: {model_path}")
    st.stop()

# ------------------------------------
# 3️⃣ Transform Setup
# ------------------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

classes = ['Not Drowsy 😎', 'Drowsy 😴']

# ------------------------------------
# 4️⃣ Streamlit App UI
# ------------------------------------
st.set_page_config(page_title="Drowsiness Detector", page_icon="😴", layout="centered")

st.markdown("<h1 style='text-align:center; color:#2b2d42;'>🧠 Drowsiness Detection System</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center; color:gray;'>Upload an eye/face image to check if you look Drowsy or Not</h4>", unsafe_allow_html=True)
st.write("---")

uploaded_file = st.file_uploader("📂 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", width=350)

    # Prediction button
    if st.button("🔍 Predict"):
        try:
            image_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                output = model(image_tensor)
                prob = torch.max(output).item() * 100
                _, predicted = torch.max(output, 1)
                label = classes[predicted.item()]

            # Display results
            if "Drowsy" in label:
                st.error(f"😴 **{label}** — Confidence: {prob:.2f}%")
                st.warning("⚠️ Alert! Drowsiness detected. Take a short break.")
            else:
                st.success(f"😎 **{label}** — Confidence: {prob:.2f}%")
                st.info("✅ You appear alert and active.")
        except Exception as e:
            st.error(f"Prediction Error: {e}")

st.write("---")
st.markdown("<p style='text-align:center; color:gray;'>Developed by Diya | Powered by PyTorch & Streamlit</p>", unsafe_allow_html=True)
