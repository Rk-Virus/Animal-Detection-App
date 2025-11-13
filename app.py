import cv2 as cv
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
from ultralytics import YOLO
from datetime import datetime
import os
import time
from playsound import playsound
import threading


def play_sound(file_path):
    threading.Thread(target=playsound, args=(file_path,), daemon=True).start()
ANIMAL_CLASSES = ['Buffalo', 'Elephant', 'Rhino', 'Zebra',
                  'Cheetah', 'Fox', 'Jaguar', 'Tiger', 'Lion', 'Panda']

ANIMAL_MODEL_PATH = r'model\best.pt'
GENERAL_MODEL_PATH = r'model\yolov8n.pt'

SAVE_FOLDER = "detections"
os.makedirs(SAVE_FOLDER, exist_ok=True)
LOG_FILE = os.path.join(SAVE_FOLDER, "detection_log.txt")


class DetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Animal")

        self.animal_model = YOLO(ANIMAL_MODEL_PATH)
        self.general_model = YOLO(GENERAL_MODEL_PATH)

        self.video_label = Label(root)
        self.video_label.pack()

        self.cap = cv.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            return

        self.cam_width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.cam_height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.root.geometry(f"{self.cam_width}x{self.cam_height}")

        self.recording = False
        self.video_writer = None
        self.record_start_time = None
        self.current_video_path = None 

        self.update_frame()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def log_detection(self, cls_name, conf, x1, y1, x2, y2, is_animal):
        """Append detection details to log file."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        record_status = "Yes" if is_animal else "No"
        video_path = self.current_video_path if is_animal and self.recording else ""
        line = f"{now}, {cls_name}, {conf:.2f}, {int(x1)},{int(y1)},{int(x2)},{int(y2)}, {record_status}, {video_path}\n"
        with open(LOG_FILE, "a") as f:
            f.write(line)

    def start_recording(self):
        if self.recording:
            return
        filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".avi"
        self.current_video_path = os.path.join(SAVE_FOLDER, filename)
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv.VideoWriter(self.current_video_path, fourcc, 20.0,
                                           (self.cam_width, self.cam_height))
        self.record_start_time = time.time()
        self.recording = True
        print(f"[INFO] Recording started: {self.current_video_path}")

    def stop_recording(self):
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            print("[INFO] Recording stopped.")
        self.recording = False
        self.current_video_path = None

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(1, self.update_frame)
            return

        animal_detected = False

        animal_results = self.animal_model.predict(frame, conf=0.5, stream=True)
        for r in animal_results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = self.animal_model.names[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0]
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                cv.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv.rectangle(frame, (int(x1), int(y1)),
                             (int(x2), int(y2)), (0, 255, 0), 2)
                cv.putText(frame, cls_name, (int(x1), int(y1) - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                self.log_detection(cls_name, conf, x1, y1, x2, y2, is_animal=True)
                animal_detected = True
                if cls_name == "Elephant":
                    playsound("lion.mp3")
                elif cls_name == "Buffalo":
                    playsound("lion.mp3")
                elif cls_name == "rhino":
                    playsound("lion.mp3")
                # elif cls_name == "Zebra":
                #     playsound("lion.mp3")
                elif cls_name == "Cheetah":
                    playsound("lion.mp3")
                # elif cls_name == "fox":
                #     playsound("lion.mp3")
                elif cls_name == "Jaguar":
                    playsound("lion.mp3")
                elif cls_name == "Tiger":
                    playsound("lion.mp3")
                elif cls_name == "Lion":
                    playsound("lion.mp3")
                # elif cls_name == "Panda":
                #     playsound("lion.mp3")

        general_results = self.general_model.predict(frame, conf=0.5, stream=True)
        for r in general_results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = self.general_model.names[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0]
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                cv.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                self.log_detection(cls_name, conf, x1, y1, x2, y2, is_animal=False)

        if animal_detected and not self.recording:
            self.start_recording()
        if self.recording:
            self.video_writer.write(frame)
            if time.time() - self.record_start_time >= 5:
                self.stop_recording()

        img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(Image.fromarray(img))
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.root.after(1, self.update_frame)

    def on_close(self):
        if self.cap:
            self.cap.release()
        if self.video_writer:
            self.video_writer.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = DetectionApp(root)
    root.mainloop()
