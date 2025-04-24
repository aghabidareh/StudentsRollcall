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