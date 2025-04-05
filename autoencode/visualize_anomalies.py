import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.model_selection import train_test_split

# Configuration
error_path = "/content/drive/MyDrive/final_project_data/reconstruction_errors.npy"
data_path = "/content/drive/MyDrive/final_project_data/final_dataset/autoencoder"
categories = ["clean_recyclable", "clean_compostable"]
img_size = 128

# Load reconstruction errors
reconstruction_errors = np.load(error_path)

# Load images
X_data = []
image_paths = []
for category in categories:
    folder_path = os.path.join(data_path, category)
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (img_size, img_size))
            img = img.astype("float32") / 255.0
            X_data.append(img)
            image_paths.append(img_path)

X_data = np.array(X_data)

# Set anomaly threshold (95th percentile)
threshold = np.percentile(reconstruction_errors, 95)

# Classify anomalies
is_anomaly = reconstruction_errors > threshold

# Visualize
def show_images(images, titles=None):
    plt.figure(figsize=(15, 5))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i])
        if titles:
            plt.title(titles[i], fontsize=8)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Show 10 anomalies
anomaly_images = X_data[is_anomaly][:10]
anomaly_titles = ["Anomaly"] * 10
print("🔎 Showing 10 detected anomalies:")
show_images(anomaly_images, anomaly_titles)

# Show 10 clean images
clean_images = X_data[~is_anomaly][:10]
clean_titles = ["Clean"] * 10
print("✅ Showing 10 clean samples:")
show_images(clean_images, clean_titles)
