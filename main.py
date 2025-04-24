import os
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from threading import Thread
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor
from torchvision import transforms

class VideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while not self.stopped:
            self.grabbed, self.frame = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

class AttendanceSystem:
    def __init__(self, students_folder="students", threshold=0.5):
        self.students_folder = students_folder
        self.threshold = threshold
        self.known_encodings = {}
        self.attendance = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained("nomic-ai/nomic-embed-vision-v1.5").to(self.device)
        self.processor = AutoProcessor.from_pretrained("nomic-ai/nomic-embed-vision-v1.5")
        self.load_student_encodings()

    def load_student_encodings(self):
        for filename in os.listdir(self.students_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                name = os.path.splitext(filename)[0]
                image_path = os.path.join(self.students_folder, filename)
                image = Image.open(image_path).convert('RGB')
                encodings = self.get_image_embedding(image)
                if encodings is not None:
                    self.known_encodings[name] = encodings
                else:
                    print(f"⚠️ Failed to process {filename}")

    def get_image_embedding(self, image):
        try:
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            return embedding
        except Exception as e:
            print(f"Error processing image: {e}")
            return None

    def process_frame(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = rgb_small_frame[y:y + h, x:x + w]
            face_pil = Image.fromarray(face_img)
            face_encoding = self.get_image_embedding(face_pil)

            if face_encoding is not None:
                name = self.recognize_face(face_encoding)
                self.update_attendance(name)
                self.draw_face_box(frame, y * 2, (x + w) * 2, (y + h) * 2, x * 2, name)

        return frame

    def recognize_face(self, face_encoding):
        similarities = []
        for name, known_encoding in self.known_encodings.items():
            similarity = np.dot(face_encoding, known_encoding) / (
                    np.linalg.norm(face_encoding) * np.linalg.norm(known_encoding)
            )
            similarities.append((name, similarity))

        if not similarities:
            return "Unknown"

        best_match = max(similarities, key=lambda x: x[1])
        if best_match[1] > self.threshold:
            return best_match[0]
        return "Unknown"