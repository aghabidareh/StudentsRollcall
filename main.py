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
    pass