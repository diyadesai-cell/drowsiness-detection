# 😴 Drowsiness Detection System

A real-time webcam-based drowsiness detection system using a custom CNN (Convolutional Neural Network) to classify eye states and alert the user with visual and sound warnings.

---

## 🎯 What it does

- Captures live video from your webcam
- Detects and tracks the eye region in real-time
- Classifies eyes as **drowsy** or **not drowsy** using a trained CNN model
- Triggers a **visual alert** and **sound alarm** when drowsiness is detected
- Useful for driver safety, late-night study sessions, or workplace alertness monitoring

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Language** | Python | Core programming language |
| **Deep Learning** | PyTorch | Building and training the CNN model |
| **Computer Vision** | OpenCV | Webcam capture, face/eye detection, real-time video processing |
| **Model Architecture** | Custom CNN (SmallCNN) | Lightweight convolutional network for binary classification |
| **GUI** | Tkinter / OpenCV window | Real-time display with alert overlay |
| **Audio Alert** | playsound / winsound | Sound alarm on drowsiness detection |
| **Data Handling** | NumPy | Image array processing |

---

## 🧠 How It Works

```
Webcam Feed
     │
     ▼
OpenCV — Face & Eye Detection
     │  locates eye region in each frame
     ▼
Preprocessing
     │  resize, normalize, convert to tensor
     ▼
SmallCNN Model (PyTorch)
     │  classifies eye state: drowsy / not drowsy
     ▼
Decision Logic
     │  tracks drowsy frames over time
     │  (avoids false alarms from a single blink)
     ▼
Alert System
     │  visual warning on screen
     │  sound alarm plays
     ▼
Live feedback loop continues
```

---

## 📂 Project Structure

```
drowsiness-detection/
│
├── vs code training/
│   ├── train_eye_cnn.py              # Trains the CNN model on eye images
│   ├── train.py                      # Training entry point
│   ├── train_smallcnn_progress.py    # Training with progress tracking
│   ├── testing.py                    # Model evaluation on test data
│   ├── evalution.py                  # Performance metrics calculation
│   ├── check_duplicates.py           # Dataset cleaning utility
│   ├── live_drowsy_detect.py         # Real-time webcam detection (main script)
│   ├── gui_drowsy_predict.py         # GUI version of the predictor
│   ├── gui_drowsy_predict_final.py   # Final GUI application
│   ├── video_drowsy_predict.py       # Run detection on a video file
│   ├── smallcnn_drowsiness(4).pth    # Trained model weights
│   └── model_metrics.csv             # Accuracy / loss results
│
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install torch torchvision opencv-python numpy playsound
```

### Run real-time detection
```bash
cd "vs code training"
python live_drowsy_detect.py
```
This opens your webcam and starts monitoring in real-time. Press `Q` to quit.

### Run the GUI version
```bash
python gui_drowsy_predict_final.py
```

### Run on a video file instead of webcam
```bash
python video_drowsy_predict.py
```

### Train the model yourself
```bash
python train_eye_cnn.py
```

---

## 📊 Model Performance

See `model_metrics.csv` for detailed accuracy and loss metrics from training.

---

## 📦 Dataset

This model was trained on a public eye-state / drowsiness detection dataset (source to be added). The dataset is organized as:

```
train_data/
├── drowsy/         # images of closed/drowsy eyes
└── notdrowsy/       # images of open/alert eyes

val_data/
├── drowsy/
└── notdrowsy/
```

> Note: The dataset is not included in this repository due to size. If you'd like to retrain the model, you'll need to source a similar drowsiness/eye-state image dataset and organize it in the structure above.

---

## 🧪 Key Concepts Demonstrated

**CNN (Convolutional Neural Network)** — learns visual patterns (eye openness, shape) directly from pixel data without manual feature engineering.

**Real-time inference** — the model processes each webcam frame fast enough to give live feedback, not just batch predictions.

**Binary classification** — the core ML task is classifying each eye frame into one of two classes: drowsy or not drowsy.

**Temporal smoothing** — rather than alerting on a single frame (which could be a normal blink), the system tracks drowsy state across multiple consecutive frames before triggering an alert, reducing false positives.

---

## 🌱 Future Improvements

- [ ] Add yawning detection as a second signal
- [ ] Track head pose / nodding for additional context
- [ ] Mobile app version
- [ ] Deploy as a desktop application (.exe)
- [ ] Add data logging for drowsiness episodes over time

---

## 👤 About

Built as a computer vision project applying deep learning to a real-world safety problem.

**Skills demonstrated:** Python · PyTorch · OpenCV · CNN · Computer Vision · Real-time Systems · Model Training & Evaluation

---

## 📄 License

MIT License — feel free to use, modify, and share.
