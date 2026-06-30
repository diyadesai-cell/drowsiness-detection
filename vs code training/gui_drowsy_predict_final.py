import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, Label, Button, messagebox
import os
import winsound  # for alert beep on Windows

# ------------------------------
# 1️⃣ Define CNN Model
# ------------------------------
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

# ------------------------------
# 2️⃣ Load Model Safely
# ------------------------------
model_path = r"C:\Users\91973\Desktop\drowsiness dataset2\vs code training\smallcnn_drowsiness(4).pth"
device = torch.device("cpu")

model = SmallCNN()
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
else:
    messagebox.showerror("Model Error", f"Model not found:\n{model_path}")
    exit()

# ------------------------------
# 3️⃣ Define Transform
# ------------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])
classes = ['notdrowsy', 'drowsy']

# ------------------------------
# 4️⃣ GUI Setup
# ------------------------------
root = tk.Tk()
root.title("😴 Drowsiness Detector - Advanced")
root.geometry("550x700")
root.configure(bg="#edf2fb")

# Header
title_label = Label(
    root, text="🧠 Drowsiness Detection System",
    font=("Segoe UI", 20, "bold"), bg="#edf2fb", fg="#2b2d42"
)
title_label.pack(pady=25)

# Frame for image preview
frame = tk.Frame(root, bg="white", bd=2, relief="groove")
frame.pack(pady=10)
img_label = Label(frame, bg="white")
img_label.pack(padx=10, pady=10)

# Result label area
result_label = Label(
    root, text="Upload an image to start prediction",
    font=("Segoe UI", 14), bg="#edf2fb", fg="#444"
)
result_label.pack(pady=30)

# ------------------------------
# 5️⃣ Prediction Function
# ------------------------------
def predict_image(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(image_tensor)
            prob = torch.max(output).item() * 100
            _, predicted = torch.max(output, 1)
            label = classes[predicted.item()]

        # If drowsy, play beep sound
        if label == "drowsy":
            winsound.Beep(1000, 800)  # frequency, duration

        # Update result label with color & emoji
        emoji = "😴" if label == "drowsy" else "😊"
        color = "#e63946" if label == "drowsy" else "#2a9d8f"
        result_label.config(
            text=f"{emoji} Prediction: {label.upper()} ({prob:.2f}%)",
            fg=color
        )
    except Exception as e:
        messagebox.showerror("Prediction Error", f"Error: {e}")

# ------------------------------
# 6️⃣ Upload Button Function
# ------------------------------
def upload_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    if not file_path:
        return

    # Display image
    img = Image.open(file_path).resize((350, 350))
    tk_img = ImageTk.PhotoImage(img)
    img_label.config(image=tk_img)
    img_label.image = tk_img

    # Run prediction
    predict_image(file_path)

# Buttons
upload_btn = Button(
    root, text="📂 Upload Image", command=upload_image,
    bg="#0077b6", fg="white", font=("Segoe UI", 12, "bold"),
    relief="flat", padx=20, pady=10
)
upload_btn.pack(pady=15)

exit_btn = Button(
    root, text="Exit", command=root.quit,
    bg="#e63946", fg="white", font=("Segoe UI", 12, "bold"),
    relief="flat", padx=20, pady=10
)
exit_btn.pack(pady=5)

root.mainloop()
