import numpy as np
import os
from tqdm import tqdm
import cv2
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, Conv2D, SeparableConv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Dense, BatchNormalization
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import tensorflow as tf


# --- Configuration ---
labels = ['NORMAL', 'PNEUMONIA']
image_size = 150
data_dirs = ["train", "val", "test"]
base_path = "chest_xray"


# --- Data Preprocessing ---
image_data, label_data = [], []

print("Pre-processing data\n")
for data_dir in data_dirs:
    for label in labels:
        data_path = os.path.join(base_path, data_dir, label)
        if not os.path.exists(data_path):
            print(f"Warning: Directory not found: {data_path}")
            continue
        for image_file in tqdm(os.listdir(data_path), desc=f"Processing {data_dir}/{label}"):
            image_path = os.path.join(data_path, image_file)
            image = cv2.imread(image_path)
            if image is None:
                continue
            image = cv2.resize(image, (image_size, image_size))
            image_data.append(image)
            label_data.append(label)

image_data = np.array(image_data) / 255.0  # Normalize
label_data = np.array(label_data)
image_data, label_data = shuffle(image_data, label_data, random_state=42)


# Train/Test Split
X_train, X_test, Y_train, Y_test = train_test_split(image_data, label_data, test_size=0.2, random_state=42)
