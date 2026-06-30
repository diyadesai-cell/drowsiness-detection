import cv2
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm

# ---------------------------------------
# 1️⃣ Eye Crop Dataset Loader
# ---------------------------------------
class EyeCropDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

        for label in os.listdir(root_dir):
            class_path = os.path.join(root_dir, label)
            for img_name in os.listdir(class_path):
                self.samples.append((os.path.join(class_path, img_name), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(gray, 1.3, 5)

        # Crop first detected eye or fallback to center crop
        if len(eyes) > 0:
            (x, y, w, h) = eyes[0]
            eye_crop = img[y:y+h, x:x+w]
        else:
            h, w, _ = img.shape
            start = h // 4
            eye_crop = img[start:start + h//2, :]

        eye_crop = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(eye_crop)

        if self.transform:
            pil_img = self.transform(pil_img)

        label_idx = 1 if label == 'drowsy' else 0
        return pil_img, label_idx


# ---------------------------------------
# 2️⃣ CNN Model
# ---------------------------------------
class EyeCNN(nn.Module):
    def __init__(self):
        super(EyeCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# ---------------------------------------
# 3️⃣ Setup
# ---------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

data_dir = r"C:\Users\91973\Desktop\drowsiness dataset2\train_data"
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dataset = EyeCropDataset(data_dir, transform=transform)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = EyeCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ---------------------------------------
# 4️⃣ Train
# ---------------------------------------
for epoch in range(10):
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in tqdm(train_loader):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()

    print(f"Epoch {epoch+1}/10 - Loss: {total_loss/len(train_loader):.4f} - Acc: {100*correct/total:.2f}%")

torch.save(model.state_dict(), "eye_cnn_drowsy.pth")
print(" Model saved as eye_cnn_drowsy.pth")
