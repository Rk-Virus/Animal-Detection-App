# Animal Detection App

A real-time animal detection and recording application using YOLOv8, OpenCV, and Tkinter. The app detects specific animals, logs detections, and records short video clips when animals are detected. It also plays alert sounds for certain animals.

---

## Features

- Real-time detection of animals using YOLOv8 models.
- Differentiates between specific animals (Elephant, Buffalo, Rhino, etc.).
- Records 5-second video clips whenever an animal is detected.
- Logs detection details (class, confidence, bounding box coordinates, recording status) to a CSV-style log file.
- Plays alert sound for select animals using `playsound`.
- Simple GUI using Tkinter to display live video feed.

---

## Setup & Requirements

### 1. Create a Virtual Environment (Recommended)
It’s best to run the app in a virtual environment to avoid conflicts:

# Windows
```python
python -m venv venv
venv\Scripts\activate
```

# macOS / Linux
```python
python3 -m venv venv
source venv/bin/activate
```
### 2. Install Dependencies
All dependencies are listed in requirements.txt. Install them inside your virtual environment:

```python
pip install -r requirements.txt
```
⚠️ Note: On Windows, playsound may require version 1.2.2 for compatibility.

### 3. Prepare Models
- Place your YOLO models in the model/ folder:

- best.pt → custom animal detection model.

- yolov8n.pt → general YOLOv8 model.

### 4. Place Audio Files
Put alert sounds (e.g., lion.mp3) in the project folder or update paths in the code accordingly.

### 5. Run the Application

With the virtual environment activated, run:

```python
    python main.py
```

- A GUI window will open showing the camera feed.
- Detected animals are highlighted with bounding boxes.
- Detection logs are saved in `detections/detection_log.txt`.
- Video clips of detected animals are saved in the `detections/` folder.

### Notes

- Recording stops automatically after 5 seconds.
- Works best with a webcam for real-time detection.
- Make sure the virtual environment is activated whenever you run the app.