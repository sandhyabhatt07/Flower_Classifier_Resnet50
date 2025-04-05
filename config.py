"""
Configuration settings for the flower classification project
"""

import os

# Data paths
DATA_ROOT = r"C:\Users\Sandhya\Deep_Learning\DL_Practice\Flower_Dataset"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
VAL_CSV = os.path.join(DATA_ROOT, "val.csv")
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR = os.path.join(DATA_ROOT, "val")
CLASS_NAMES_PATH = os.path.join(DATA_ROOT, "classname.txt")


# Model settings
NUM_CLASSES = 14  # Update this based on your dataset
MODEL_SAVE_PATH = "checkpoints/flower_classifier.pth"
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001

# Image settings
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]  # ImageNet mean
STD = [0.229, 0.224, 0.225]   # ImageNet std

# Web app settings
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}