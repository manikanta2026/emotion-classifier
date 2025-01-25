# Speech Emotion Recognition Model

## Overview
A deep learning model for recognizing emotions from audio signals using multiple datasets and advanced feature extraction techniques.

## Datasets
-TESS
-RAVDESS
-CREMA
-SAVEE
-Link to Datasets:-https://www.kaggle.com/manikantagade/datasets


## Features
- Multi-dataset emotion recognition (TESS, RAVDESS, CREMA, SAVEE)
- Advanced feature extraction with librosa
- Artificial Neural Network (ANN) architecture
- Data augmentation techniques
- Comprehensive preprocessing and scaling

## Key Technologies
- Python
- NumPy
- TensorFlow/Keras
- Librosa
- Scikit-learn
- Matplotlib

## Model Architecture
- 4-layer Dense Neural Network
- Regularization (L2)
- Batch Normalization
- Dropout for preventing overfitting
- Softmax output layer

## Supported Emotions
- Neutral
- Happy
- Sad
- Angry
- Fear
- Disgust

## Feature Extraction
Extracted features include:
- MFCCs
- Mel Spectrogram
- Spectral Centroids
- Tempo
- RMS Energy
- Chroma Features
- Zero Crossing Rate

## Data Augmentation Techniques
- Noise injection
- Feature shifting
- Scaling
- Random inversion

## Performance Metrics
- Categorical Cross-Entropy Loss
- Accuracy

## Usage
1. Preprocess audio files
2. Extract features
3. Scale features
4. Train/evaluate model

## Prerequisites
- Python 3.8+
- TensorFlow
- Librosa
- NumPy
- Scikit-learn

## Model Training
- Epochs: 150
- Batch Size: 64
- Validation Split: 20%

## Authors
G Manikanta

