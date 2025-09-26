# AI Models for Parkinson’s Detection and Image Classification

This repository contains two Jupyter notebooks that implement machine learning and deep learning pipelines for medical and image-based classification tasks:

parkinsons-test-ai.ipynb – Parkinson’s Disease Detection using motion/handwriting test data.

ImageClassification_CNN.ipynb – Image classification (Healthy vs. Patient / Happy vs. Sad) using Convolutional Neural Networks (CNNs).

├── parkinsons-test-ai.ipynb   # Parkinson’s disease detection (ML models)
├── ImageClassification_CNN.ipynb  # CNN-based image classifier
├── control                       # Dataset for normal patient
|── parkinson                     # Dataset for parkinson patient
└── README.md                  # Project documentation

# ⚙️ Requirements

Both notebooks were developed in Python 3.x with the following key libraries:

Core Libraries: numpy, pandas, matplotlib, seaborn

Machine Learning: scikit-learn

Deep Learning: tensorflow, keras

Image Processing: cv2 (OpenCV), imghdr

Utilities: os, time

pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras opencv-python

# 🧩 Notebook 1: Parkinson’s Disease Detection (parkinsons-test-ai.ipynb)
# 🔹 Problem Statement

Detect Parkinson’s Disease from handwriting/motion test data by extracting meaningful features such as stroke counts, velocity, acceleration, jerk, and pressure patterns.

# 🔹 Workflow

Data Input: Reads handwriting test data from .csv or dataset folder.

# Feature Engineering:

Number of strokes

Speed, velocity, acceleration, jerk

In-air vs. on-surface time

NCV (Normalized Cumulative Velocity), NCA (Normalized Cumulative Acceleration)

# Dataset Creation:

Combines extracted features into structured data.csv

Assigns labels: 1 (Parkinson’s patient), 0 (Control group)

# Model Training:

ML classifiers: Logistic Regression, Decision Tree, Random Forest, KNN, SVM

# Evaluation:

Accuracy calculation

Confusion matrix, precision/recall metrics

# Visualization:

Seaborn plots of handwriting pressure over time

Comparison between Parkinson’s vs. Control samples

# 🧩 Notebook 2: Image Classification (ImageClassification_CNN.ipynb)
# 🔹 Problem Statement

Classify images (e.g., Healthy vs. Patient or Happy vs. Sad) using a Convolutional Neural Network (CNN).

# 🔹 Workflow

# Dataset Handling:

Images stored in directories (Datasets/Healthy, Datasets/Patient)

tf.keras.utils.image_dataset_from_directory() used for loading

Removes “dodgy”/corrupted images using cv2 + imghdr

# Data Preprocessing:

Resizing images to 256x256

Normalization (/255.0)

Splitting into Train (70%), Validation (20%), Test (10%)

# Model Architecture:

Convolutional + MaxPooling layers

Dense layers with dropout

Binary classification output

# Training:

Optimizer: Adam

Loss: Binary Cross-Entropy

Metrics: Accuracy

# Evaluation:

Accuracy and loss plots

Precision, Recall, Binary Accuracy

Testing on External Images:

Load new image with OpenCV

Preprocess (resize, normalize)

Predict class (Healthy / Patient)

# 📊 Results

# Parkinson’s Notebook:

Extracted motion features allow ML models to achieve high accuracy in distinguishing Parkinson’s patients vs. control group.

Logistic Regression, Random Forest, and SVM provided competitive results.

# Image Classification Notebook:

CNN achieved high training/validation accuracy after ~10 epochs.

External image testing pipeline included for real-world inference.

