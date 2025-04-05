import numpy as np
from tensorflow.keras.models import load_model
from tqdm import tqdm
import cv2
import os

# Paths
model_path = "autoencoder/model_autoencoder.h5"
data_path = "data/autoencoder/clean_images"
error_save_path = "autoencoder/reconstruction_errors.npy"

# Load model
autoencoder = load_model(model_path)

# Load and preprocess images
img_size = 128
X_data = []

for fname in tqdm(os.listdir(data_path), desc="Loading clean images"):
    fpath = os.path.join(data_path, fname)
    img = cv2.imread(fpath)
    if img is not None:
        img = cv2.resize(img, (img_size, img_size))
        img = img.astype("float32") / 255.0
        X_data.append(img)

X_data = np.array(X_data)

# Predict and calculate reconstruction error
X_pred = autoencoder.predict(X_data)
reconstruction_errors = np.mean(np.square(X_data - X_pred), axis=(1, 2, 3))

# Save errors
np.save(error_save_path, reconstruction_errors)
print("✅ Reconstruction errors saved to:", error_save_path)
