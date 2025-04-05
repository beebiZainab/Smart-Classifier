# Smart-Classifier
# 🧱 Smart Recycle Classifier

An AI-powered system that automates the classification and filtering of waste materials into **Recyclable**, **Non-Recyclable**, and **Compostable** categories. It combines the strengths of **Autoencoders** for anomaly detection and **CNNs** for classification to enhance waste segregation in smart city environments.

---

## ✨ Features

- 💡 **Hybrid Deep Learning System**:
  - **Autoencoder**: Trained on clean recyclable and compostable images to detect anomalies (e.g., contaminated or odd waste).
  - **CNN Classifier**: Classifies waste into three categories after anomaly filtering.

- 📊 **Performance Optimized**:
  - CNN model trained on 10,859 training images.
  - Achieved **75% accuracy** on the test set with strong precision-recall for recyclable and non-recyclable classes.

- 🚀 **Real-time Prediction UI**:
  - Streamlit-based app for uploading and classifying waste images.
  - Outputs predicted class and confidence score.

- ♻️ **Societal Impact**:
  - Helps municipalities, environmental agencies, and citizens in better waste management and recycling.

---

## 📁 Project Structure
```
Smart-Classifier/
├── README.md
├── requirements.txt
├── .gitignore
├── streamlit_app.py           # Streamlit UI script
├── autoencoder_model.h5       # Trained autoencoder model
├── cnn_classifier_model.h5     # Trained CNN classifier
├── reconstruction_errors.npy  # Saved error array from autoencoder
├── utils/
│   └── preprocessing.py        # Image filtering and dataset creation
├── final_dataset_split/
│   ├── train/val/test/       # Classified dataset split
└── final_dataset/autoencoder/
    ├── clean_recyclable/
    └── clean_compostable/
```

---

## 📖 How It Works

1. **Data Preparation**:
   - Started with the TrashNet dataset.
   - Clean images filtered for autoencoder training using OpenCV thresholding.
   - Reclassified and split all images for CNN into three categories.

2. **Autoencoder Training**:
   - Trained on 6,093 clean images (recyclable + compostable).
   - Used MSE reconstruction loss.
   - Images with high reconstruction error flagged as anomalies.

3. **CNN Classification**:
   - Trained on structured dataset:
     - Train: 10,859
     - Validation: 2,328
     - Test: 2,328
   - Achieved macro F1 score of ~67%.

4. **Deployment**:
   - Deployed via Streamlit.
   - Option to integrate with Flask or other interfaces for future scalability.

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/beebiZainab/Smart-Classifier.git
cd Smart-Classifier
```

### 2. Install requirements
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App
```bash
streamlit run streamlit_app.py
```

---

## 📊 Results

- **Classification Accuracy**: ~75%
- **Anomaly Detection**: Detected top 5% anomalous images using reconstruction error thresholding.

| Class          | Precision | Recall | F1 Score |
|----------------|-----------|--------|----------|
| Recyclable     | 0.65      | 0.81   | 0.72     |
| Compostable    | 0.50      | 0.48   | 0.49     |
| Non-Recyclable | 0.87      | 0.74   | 0.80     |


---

## 🌿 Business/Societal Value

- 🛋️ **Municipal Bodies**: Enhance sorting efficiency at waste collection points.
- 🛠️ **Recyclers**: Reduce contamination and get cleaner feedstock.
- 🌼 **NGOs & Educators**: Drive behavioral change through visibility and gamification.

---

## 💚 Acknowledgments

- [TrashNet Dataset](https://github.com/garythung/trashnet)
- TensorFlow & Keras
- Streamlit for Web App

---

## 📜 License
This project is licensed under the MIT License. See the LICENSE file for details.

---

## 📢 Contact
**Zainab Beebi**
- GitHub: [@beebiZainab](https://github.com/beebiZainab)
- Email: beebizainab29@gmail.com


