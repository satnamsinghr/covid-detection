# COVID-19 Detection using Convolutional Neural Networks (CNN)

**Repository:** COVID-Detection \
**Author:** Satnam Singh Ramavath \
**Contact:** satnamsinghramavath9320@gmail.com\
**GitHub:** https://github.com/satnamsinghr

---

## Project Summary

This project implements a deep learning-based CNN model to detect COVID-19 infections from chest X-ray images. By leveraging convolutional layers and careful preprocessing, the model distinguishes between COVID-19, pneumonia, and normal cases with high accuracy.

The work demonstrates the potential of AI in medical imaging, especially during pandemic situations where rapid diagnosis is crucial.

---

## Motivation

- COVID-19 caused a global crisis where fast and reliable diagnostic tools became essential.
- Traditional RT-PCR tests were time-consuming and not always accessible.
- X-ray imaging is widely available and cheaper, but manual interpretation is slow.
- Using CNNs, we automate X-ray image analysis for faster and more accurate COVID detection.

---

## What’s in this repository

- Notebooks/COVID_Detection.ipynb — main Colab notebook: preprocessing, model training, evaluation, visualization.
- papers/covid_detection_paper.pdf — project documentation / report.
- requirements.txt — dependencies required to reproduce the results.
- screenshots/ — sample predictions, confusion matrix, training curves.
- README.md — this documentation file.

---

## Dataset

We used publicly available Chest X-ray datasets that contain images for COVID-19, pneumonia, and healthy lungs. The dataset is not uploaded here due to size. Instead, use the following options in Colab:

Download with Kaggle API:

!pip install -q kaggle

---

## Upload kaggle.json (from Kaggle account)

!kaggle datasets download -d -p /content/data --unzip
Or, load from Google Drive:
from google.colab import drive drive.mount('/content/drive')

---

## copy dataset from Drive into /content/data 

### Preprocessing

- Convert all X-ray images to grayscale / RGB normalized format.
- Resize images to fixed dimensions (e.g., 224x224).
- Data augmentation: rotation, flip, brightness adjustments.
- Split into train / validation / test sets.

---

## Model Architecture

- Input layer: resized 224x224 X-ray images.
- Convolutional + MaxPooling layers: to extract features.
- Flatten → Fully Connected layers.
- Dropout for regularization.
- Output layer: Softmax classifier for COVID / Pneumonia / Normal.
- Alternative architectures like ResNet, VGG, and custom CNNs can also be tested.

---
## Training Details

- Optimizer: Adam
- Loss: Categorical Cross-Entropy
- Metrics: Accuracy, Precision, Recall, F1-score
- Early stopping and checkpointing used to avoid overfitting.
- Training performed on Google Colab GPU runtime.

---
## Evaluation Metrics

- Accuracy, Precision, Recall, F1-score
- Confusion Matrix for class-level analysis
- ROC and AUC for binary/multi-class performance
- Expected outcome: high accuracy (~95%+) for COVID vs Non-COVID classification.

---
## How to Run 
### Run on Google Colab

1. Open the notebook:
2. Runtime → Change runtime type → GPU.
3. Install required packages (first cells handle it).
4. Download dataset (Kaggle API or Drive).
5. Run all cells → model will train and evaluate.

---
## Run locally

Clone repo:
git clone https://github.com/satnamsinghr/COVID-Detection.git
cd COVID-Detection
Create virtual environment & install requirements:
python -m venv venv source venv/bin/activate # Linux/Mac venv\Scripts\activate # Windows pip install -r requirements.txt
Run Jupyter notebook:
jupyter notebook notebooks/COVID_Detection.ipynb

---

## Folder Structure

COVID-Detection/ ├── notebooks/ │ └── COVID_Detection.ipynb ├── papers/ │ └── covid_detection_paper.pdf ├── screenshots/ │ ├── training_curve.png │ ├── confusion_matrix.png │ └── predictions.png ├── requirements.txt └── README.md

---

## Limitations

- Dataset size is limited compared to real-world diversity.
- Performance depends on dataset quality and balance.
- This project is for research/academic demonstration only — not a medical diagnostic tool.

---

## Future Work

- Use larger and more diverse datasets.
- Apply transfer learning with ResNet/VGG/DenseNet.
- Hyperparameter tuning for robustness.
- Deploy as a simple web app (Flask/FastAPI + Docker).
- Experiment with Grad-CAM for explainability (heatmaps).

---
## References

- Dataset sources: Kaggle Chest X-ray datasets.
- CNN architectures: Deep Learning literature (ResNet, VGG).
- Related medical imaging research papers.

---
## Author & Contact
Satnam Singh Ramavath
- Email: satnamsinghramavath9320@gmail.com
- GitHub: satnamsinghr
