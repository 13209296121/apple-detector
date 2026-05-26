# -*- coding: utf-8 -*-
"""
Apple Detector App (ONNX version)
Receives ESP32-CAM video stream and detects apples in real time using ONNX Runtime.
"""

import cv2
import numpy as np
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image as KivyImage
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import threading
import time
import os
from datetime import datetime

# ================== Configuration ==================
STREAM_URL = "http://192.168.153.201/stream"
MODEL_PATH = "yolov8n.onnx"
CONF_THRESHOLD = 0.35
SAVE_DIR = "apple_captures"
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]
# ===================================================

class AppleDetector(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', **kwargs)
        
        self.image = KivyImage(size_hint=(1, 0.9))
        self.add_widget(self.image)
        
        self.info_label = Label(
            text="Initializing...",
            size_hint=(1, 0.05),
            font_size='14sp',
            color=(0, 1, 0, 1)
        )
        self.add_widget(self.info_label)
        
        btn_layout = BoxLayout(size_hint=(1, 0.1), spacing=10, padding=10)
        self.start_btn = Button(text="Start")
        self.start_btn.bind(on_press=self.start_detection)
        btn_layout.add_widget(self.start_btn)
        
        self.capture_btn = Button(text="Capture", disabled=True)
        self.capture_btn.bind(on_press=self.save_screenshot)
        btn_layout.add_widget(self.capture_btn)
        
        self.stop_btn = Button(text="Stop")
        self.stop_btn.bind(on_press=self.stop_detection)
        btn_layout.add_widget(self.stop_btn)
        
        self.add_widget(btn_layout)
        
        self.running = False
        self.session = None
        self.cap = None
        self.current_frame = None
        self.fps_timer = time.time()
        self.frame_count = 0
        self.fps = 0
        os.makedirs(SAVE_DIR, exist_ok=True)

    def start_detection(self, instance):
        if self.running:
            return
        self.running = True
        self.info_label.text = "Loading model..."
        self.start_btn.disabled = True
        
        threading.Thread(target=self._init_and_run, daemon=True).start()
        Clock.schedule_interval(self.update_frame, 1/30.0)

    def _init_and_run(self):
        try:
            import onnxruntime as ort
            self.session = ort.InferenceSession(MODEL_PATH)
            input_details = self.session.get_inputs()
            output_details = self.session.get_outputs()
            input_name = input_details[0].name
            input_size = input_details[0].shape[2:4]  # (height, width)
            self.info_label.text = "Model loaded (ONNX)"

            self.cap = cv2.VideoCapture(STREAM_URL)
            if not self.cap.isOpened():
                self.info_label.text = "Cannot connect to camera. Check hotspot and IP."
                self.running = False
                return
            self.info_label.text = "Detection started"
            self.capture_btn.disabled = False
            
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.5)
                    continue
                
                # Preprocess: BGR → RGB, resize, normalize, channel transpose
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img, (input_size[1], input_size[0]))
                img_norm = np.expand_dims(img_resized, axis=0).astype(np.float32) / 255.0
                img_norm = np.transpose(img_norm, (0, 3, 1, 2))  # (1, H, W, C) → (1, C, H, W)

                # Run ONNX inference
                outputs = self.session.run([output_details[0].name], {input_name: img_norm})
                outputs = outputs[0]  # (1, 84, 8400)
                outputs = np.squeeze(outputs)  # (84, 8400)

                annotated = frame.copy()
                apple_count = 0
                h, w = frame.shape[:2]
                
                # Parse YOLOv8 ONNX output
                for i in range(outputs.shape[1]):
                    det = outputs[:, i]
                    class_scores = det[4:]
                    class_id = int(np.argmax(class_scores))
                    score = float(class_scores[class_id])
                    if score < CONF_THRESHOLD or COCO_CLASSES[class_id] != 'apple':
                        continue
                    
                    # Center x, y, width, height (normalized)
                    cx, cy, bw, bh = det[0], det[1], det[2], det[3]
                    x1 = int((cx - bw / 2) * w)
                    y1 = int((cy - bh / 2) * h)
                    x2 = int((cx + bw / 2) * w)
                    y2 = int((cy + bh / 2) * h)
                    
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    apple_count += 1
                
                cv2.putText(annotated, f"Apples: {apple_count}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                self.frame_count += 1
                if self.frame_count % 10 == 0:
                    elapsed = time.time() - self.fps_timer
                    self.fps = 10 / elapsed if elapsed > 0 else 0
                    self.fps_timer = time.time()
                    self.info_label.text = f"Apples: {apple_count} | FPS: {self.fps:.1f}"
                
                self.current_frame = annotated
                
        except Exception as e:
            self.info_label.text = f"Error: {str(e)}"
            self.running = False

    def update_frame(self, dt):
        if self.current_frame is not None and self.running:
            buf = cv2.flip(self.current_frame, 0).tobytes()
            texture = Texture.create(
                size=(self.current_frame.shape[1], self.current_frame.shape[0]),
                colorfmt='bgr'
            )
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.image.texture = texture

    def save_screenshot(self, instance):
        if self.current_frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{SAVE_DIR}/apple_{timestamp}.jpg"
            cv2.imwrite(filename, self.current_frame)
            self.info_label.text = f"Saved: {filename}"

    def stop_detection(self, instance):
        self.running = False
        Clock.unschedule(self.update_frame)
        if self.cap:
            self.cap.release()
        self.info_label.text = "Stopped"
        self.start_btn.disabled = False
        self.capture_btn.disabled = True

class AppleApp(App):
    def build(self):
        self.title = "Apple Detector (ONNX)"
        return AppleDetector()
    
    def on_stop(self):
        if hasattr(self, 'root') and self.root.running:
            self.root.stop_detection(None)

if __name__ == "__main__":
    AppleApp().run()