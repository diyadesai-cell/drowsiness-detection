import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import Label
import winsound
import threading
import os

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
# 2️⃣ Load Trained Model
# ------------------------------
model_path = r"C:\Users\91973\Desktop\drowsiness dataset2\vs code training\smallcnn_drowsiness(4).pth"
device = torch.device("cpu")

model = SmallCNN().to(device)
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("✅ Model loaded successfully")
else:
    print("❌ Model file not found!")
    exit()

# ------------------------------
# 3️⃣ Transform Setup
# ------------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])
classes = ["Not Drowsy 😎", "Drowsy 😴"]

# ------------------------------
# 4️⃣ Helper – Beep in Background
# ------------------------------
def play_beep():
    threading.Thread(target=lambda: winsound.Beep(1000, 700)).start()

# ------------------------------
# 5️⃣ Tkinter Window
# ------------------------------
root = tk.Tk()
root.title("🧠 Live Drowsiness Detection")
root.geometry("850x680")
root.configure(bg="#edf2fb")

title_label = Label(root, text="😴 Real-Time Drowsiness Detection", 
                    font=("Segoe UI", 20, "bold"), bg="#edf2fb", fg="#2b2d42")
title_label.pack(pady=20)

video_label = Label(root)
video_label.pack()

result_label = Label(root, text="Starting camera...", font=("Segoe UI", 16), bg="#edf2fb")
result_label.pack(pady=15)

# ------------------------------
# 6️⃣ Camera + Detection Setup
# ------------------------------
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if not cap.isOpened():
    result_label.config(text="❌ Cannot access camera.")
    raise SystemExit

frame_skip = 5
frame_count = 0
drowsy_counter = 0
alert_threshold = 5

def update_frame():
    global frame_count, drowsy_counter

    ret, frame = cap.read()
    if not ret:
        result_label.config(text="Camera disconnected!")
        return

    frame_count += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    label_text = "No Face Detected"
    prob = 0

    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        face_crop = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

        # Try to detect eyes inside face
        eyes = eyes_cascade.detectMultiScale(cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY), 1.3, 5)
        if len(eyes) >= 1:
            (ex, ey, ew, eh) = eyes[0]
            eye_crop = face_crop[ey:ey+eh, ex:ex+ew]
        else:
            eye_crop = face_crop  # fallback to full face

        # Predict every few frames
        if frame_count % frame_skip == 0:
            img = Image.fromarray(cv2.cvtColor(eye_crop, cv2.COLOR_BGR2RGB))
            image_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(image_tensor)
                _, predicted = torch.max(output, 1)
                prob = torch.max(output).item() * 100
                label_text = classes[predicted.item()]

            if "Drowsy" in label_text:
                drowsy_counter += 1
            else:
                drowsy_counter = 0

            if drowsy_counter >= alert_threshold:
                play_beep()
                result_label.config(
                    text=f"⚠️ Drowsy Detected ({prob:.2f}%)",
                    fg="#e63946", font=("Segoe UI", 18, "bold")
                )
            elif "Drowsy" in label_text:
                result_label.config(
                    text=f"😴 Slight Drowsiness ({prob:.2f}%)",
                    fg="#fb8500", font=("Segoe UI", 16, "bold")
                )
            else:
                result_label.config(
                    text=f"✅ Not Drowsy ({prob:.2f}%)",
                    fg="#2a9d8f", font=("Segoe UI", 16, "bold")
                )

    else:
        result_label.config(text="👀 No face detected", fg="#6c757d")

    # Show frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(rgb_frame, (700, 500))
    tk_img = ImageTk.PhotoImage(Image.fromarray(frame_resized))
    video_label.config(image=tk_img)
    video_label.image = tk_img

    root.after(10, update_frame)


# ------------------------------
# 7️⃣ Exit Function
# ------------------------------
def on_close():
    cap.release()
    cv2.destroyAllWindows()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)
update_frame()
root.mainloop()
