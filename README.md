# Building Fake Bank Notes Detection System Using Deep Learning (ANN)

## Overview
Counterfeit banknotes pose a serious threat to financial systems worldwide. This project focuses on detecting fake banknotes using **Artificial Neural Networks (ANN)**. The model classifies banknotes as **Genuine** or **Fake** based on statistical features extracted from banknote images using wavelet transformation techniques.

## Objective
The primary objective of this project is to build an accurate and reliable deep learning model that can help banks, financial institutions, and automated systems (such as ATMs) detect counterfeit currency and reduce financial fraud.

## Dataset Details
The dataset used in this project is the **Banknote Authentication Dataset**, which is publicly available on Kaggle.

### Features
- **Variance (VWTI)** – Variance of Wavelet Transformed Image  
- **Skewness (SWTI)** – Skewness of Wavelet Transformed Image  
- **Curtosis (CWTI)** – Curtosis of Wavelet Transformed Image  
- **Entropy (EI)** – Entropy of Image  
- **Class** – Target variable (1 = Genuine, 0 = Fake)

## Project Workflow
1. **Data Preprocessing**
   - Load and explore the dataset
   - Feature scaling using `StandardScaler`
   - Split data into training,validation and testing sets

2. **Model Development**
   - Build an Artificial Neural Network (ANN) using TensorFlow/Keras
   - Train the model on the training dataset
   - Evaluate the model using accuracy, precision, recall, and F1-score

3. **Prediction System**
   - Accept new banknote feature values as input
   - Apply preprocessing and scaling
   - Predict whether the banknote is genuine or fake

## Model Performance
The ANN model achieves **high accuracy** in detecting fake and genuine banknotes. Performance may vary slightly depending on the train-test split and model configuration.

## Installation and Setup

### Clone the Repository
```bash
git clone https://github.com/surisettyrahul/Building-Fake-Bank-Notes-Detection-System-Using-Deep-learning-ANN.git
cd Building-Fake-Bank-Notes-Detection-System-Using-Deep-learning-ANN
```
### Install Dependencies
```bash
pip install -r requirements.txt
```
### Run the project
```bash
streamlit run app.py
```
