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