# 🫁 Chest X-Ray (Pneumonia) Image Classification

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Project Overview

This project aims to classify chest X-ray images into two categories: **Normal** and **Pneumonia**, using deep learning. The goal is to build an automated diagnostic tool that can assist medical professionals in detecting pneumonia from radiographic images. This is accomplished by implementing and comparing three different deep learning approaches using Convolutional Neural Networks (CNNs).

The project uses a publicly available dataset of pediatric chest X-ray images and applies three distinct modeling strategies:

1.  **Custom CNN:** A baseline model built from scratch.
2.  **Transfer Learning (Feature Extraction):** Using a pre-trained VGG16 model as a fixed feature extractor.
3.  **Fine-Tuning:** Unfreezing the last layers of the pre-trained VGG16 model to adapt it specifically to the medical imaging task.

---

## 📊 Dataset Information

The dataset used is the **Chest X-Ray Images (Pneumonia)** dataset (version 3), which contains 5,856 validated chest X-ray images.

- **Source:** [Labeled Chest X-Ray Images on Kaggle](https://www.kaggle.com/tolgadincer/labeled-chest-xray-images)
- **Image Classes:**
  - `NORMAL`: Healthy lung X-ray images (1,585 samples).
  - `PNEUMONIA`: X-ray images showing signs of pneumonia (4,273 samples).
- **Data Split:**
  - **Training Set:** 5,232 images (used for training).
  - **Testing Set:** 624 images (independent set for final evaluation).
  - **Validation Set:** Created by splitting the training set to tune the models during training.
- **Subject Details:** The images are from retrospective cohorts of pediatric patients (one to five years old) from the Guangzhou Women and Children’s Medical Center.

The dataset is slightly imbalanced, with the `PNEUMONIA` class having more samples.

---

## 🛠️ Methodology & Models

The project compares three different deep learning approaches:

### 1. Simple CNN (from scratch)
A custom-built Convolutional Neural Network with a few convolutional and pooling layers, followed by dense layers. This serves as a baseline to understand the complexity of the problem.

### 2. Transfer Learning (Feature Extraction)
Uses a pre-trained VGG16 model as a fixed feature extractor. The pre-trained weights are frozen, and only a new custom classifier head (dense layers) is trained on top. This leverages general image features learned from large datasets like ImageNet.

### 3. Fine-Tuning
Starts from the pre-trained VGG16 model. Instead of freezing all layers, the weights of the last few layers are "unfrozen" and re-trained jointly with the new classifier head. This allows the model to adapt its higher-level features specifically to the medical imaging domain, often leading to a performance boost.

---

## 📈 Results & Performance Metrics

The notebook evaluates and compares the performance of all three models on the independent test set (624 images).

| Model | Accuracy | Notes |
| :--- | :--- | :--- |
| Custom CNN | ~80% | Baseline model, lower performance due to small dataset |
| Transfer Learning (VGG16) | ~85% | Significant improvement over custom CNN |
| Fine-Tuning (VGG16) | ~92% | Best performance, model adapts well to medical images |

**Key Observations:**
- The fine-tuned VGG16 model achieved the best performance at approximately **92% accuracy**, confirming the effectiveness of adapting pre-trained features to the medical imaging domain.
- The imbalanced dataset (more Pneumonia samples) likely impacted the precision/recall trade-off, with higher recall for the majority class.

