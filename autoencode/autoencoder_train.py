import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

img_size = 128
data_path = "final_dataset/autoencoder"
categories = ["clean_recyclable", "clean_compostable"]

X_data = []
for category in categories:
    folder_path = os.path.join(data_path, category)
    for img_name in tqdm(os.listdir(folder_path), desc=f"Loading {category}"):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (img_size, img_size))
            img = img.astype("float32") / 255.0
            X_data.append(img)

X_data = np.array(X_data)
X_train, X_val = train_test_split(X_data, test_size=0.2, random_state=42)
