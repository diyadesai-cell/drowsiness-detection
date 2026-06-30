import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# ====== 1️⃣ Define CNN model (must match training) ======
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
            nn.Linear(8 * 8 * 64, 128),  # for 64x64 input images
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)  # 2 classes: drowsy / notdrowsy
        )

    def forward(self, x):
        return self.fc(self.conv(x))


# ====== 2️⃣ Load trained model ======
model_path = r"C:\Users\91973\Desktop\drowsiness dataset2\vs code training\smallcnn_drowsiness(3).pth"

model = SmallCNN()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

print("Model loaded successfully!")

# ====== 3️⃣ Define image transforms ======
transform = transforms.Compose([
    transforms.Resize((64, 64)),   # must match training image size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])


# ====== 4️⃣ Function to test one image ======
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # add batch dimension

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(probs, 1)
    
    classes = ['drowsy', 'notdrowsy']  # same as your training order
    print(f"\nImage: {image_path}")
    print(f"Prediction: {classes[predicted.item()]}")
    print(f"Probabilities: {probs.numpy()}\n")


# ====== 5️⃣ Test few images ======
predict_image(r"C:\Users\91973\Desktop\drowsiness dataset2\train_data\drowsy\001_glasses_sleepyCombination_600_drowsy.jpg")
predict_image(r"C:\Users\91973\Desktop\drowsiness dataset2\train_data\notdrowsy\001_glasses_nonsleepyCombination_53_notdrowsy.jpg")
