# 👶 Baby Cry Analyzer

A machine learning project designed to help parents better understand their baby's needs by analyzing crying sounds. The system uses audio signal processing and machine learning techniques to classify different types of baby cries — such as **hunger**, **discomfort**, **tiredness**, **belly pain**, or **burping** — and provide actionable insights.

---

## 🔍 Project Overview

This project processes baby crying audio data, extracts relevant features, trains various machine learning models (including deep learning models), and predicts the cause of crying. It includes a user interface for real-time analysis and visualization.

---

## 🧠 Technologies & Models Used

- **WaveNet**: Used for data augmentation and high-quality audio processing.
- **Random Forest**: Used for initial classification.
- **Conv1D-LSTM**: Used for deep learning classification.
- **SMOTE**: Applied to balance training data.
- **Tkinter GUI**: For a simple user interface.
- **noisereduce**: For cleaning audio samples.

---

## 🎧 Audio Preprocessing

- Audio formats supported: `.wav`, `.mp3`, `.ogg`
- Resampled all audio to **16kHz**
- Trimmed or padded audio to **3 seconds**
- Saved processed files to `processed_sounds/`

---

## 🔬 Feature Extraction

Features extracted include:

- MFCC (Mel-Frequency Cepstral Coefficients)  
- Spectral Centroid  
- Spectral Bandwidth  
- Spectral Rolloff  
- Zero Crossing Rate  
- RMS Energy  
- Chroma STFT

Features and labels are saved to a `.csv` file for model training.

---

## 🏗️ Data Flow

1. **Noise Reduction** using `noisereduce`
2. **Data Augmentation** using **WaveNet**
3. **Splitting**: 80% training / 20% testing
4. **Feature Extraction**
5. **Model Training** (Random Forest & Conv1D-LSTM)
6. **Evaluation** using accuracy, confusion matrix, and classification report
7. **Real-time Prediction** with pre-trained model

---

## 📊 Model Evaluation

- Metrics: **Accuracy**, **F1-Score**, **Confusion Matrix**
- Visualization of training/validation accuracy & loss
- Histogram of predicted classes

---

## 🖥️ GUI Features

- Upload audio files
- Predict crying reason in real-time
- Display result with visual feedback
- Handles errors and invalid input

---

## 🧠 Team Members

- **Hana Mohamed Gohar**  
- **Eman Mousa Kmar**  
- **Eman Hussin Butie**

---

## 📁 Folder Structure

```bash
📂 clean_baby_sounds/
📂 processed_sounds/
📁 feature_data.csv
📁 train_model.py
📁 predict_gui.py
📁 wavenet_augmentation.py
📁 feature_extraction.py
📁 model_evaluation.py
