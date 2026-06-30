import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, accuracy_score,
    precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import pandas as pd

# ------------------------
# # ------------------------
# 1. Config
# ------------------------
# CHANGE THIS LINE: Set data_dir to the folder containing train_data and val_data
data_dir = r"C:\Users\91973\Desktop\drowsiness dataset2"
batch_size = 32
model_path = "smallcnn_drowsiness(4).pth" # Note: This path is relative to the script location

# ------------------------
# 2. Transform
# ------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# ------------------------
# 3. Dataset & Dataloader
# ------------------------
train_data = datasets.ImageFolder(os.path.join(data_dir, "train_data"), transform=transform)
val_data   = datasets.ImageFolder(os.path.join(data_dir, "val_data"), transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
val_loader   = DataLoader(val_data, batch_size=batch_size, shuffle=False)
classes = train_data.classes

# ------------------------
# 4. Model Definition (must match training code)
# ------------------------
class SmallCNN(nn.Module):
    def __init__(self, num_classes):
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
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

# ------------------------
# 5. Load Model
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SmallCNN(num_classes=len(classes)).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print(" Model loaded successfully!")

# ------------------------
# ------------------------
# 6. Evaluation Function
# ------------------------
def evaluate_model(dataloader, split_name="Validation"):
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # EMOJI REMOVED HERE
    print(f"\n {split_name} Metrics:") 
    print(f"Accuracy : {acc*100:.2f}%")
    print(f"Precision: {prec*100:.2f}%")
    print(f"Recall   : {rec*100:.2f}%")
    print(f"F1-Score : {f1*100:.2f}%")

    return [acc, prec, rec, f1], y_true, y_pred


# ------------------------
# 7. Run Evaluation
# ------------------------
train_metrics, y_true_train, y_pred_train = evaluate_model(train_loader, "Training")
val_metrics, y_true_val, y_pred_val = evaluate_model(val_loader, "Validation")

# ------------------------
# ------------------------
# 8. Summary Table
# ------------------------
# ... (rest of the code) ...
print("\n Final Metrics Summary Table:") # EMOJI REMOVED HERE
print(metrics_df.round(2))

# (Optional) Save table
metrics_df.to_csv("model_metrics.csv", index=False)
print("\nMetrics saved to model_metrics.csv")


# ------------------------
# 9. Confusion Matrix (Validation)
# ------------------------
print("\nClassification Report (Validation):")
print(classification_report(y_true_val, y_pred_val, target_names=classes))

cm = confusion_matrix(y_true_val, y_pred_val)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix - Validation Data")
plt.show()
