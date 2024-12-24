import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dropout, LSTM, Dense, BatchNormalization, Bidirectional
from collections import Counter
import matplotlib.pyplot as plt
import librosa
import librosa.display
import matplotlib.pyplot as plt

data_file = r'train_data.csv'
data = pd.read_csv(data_file)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

joblib.dump(encoder, r'label_encoder.pkl')

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, r'scaler.pkl')

smote = SMOTE(k_neighbors=1, random_state=42)  
X_resampled, y_resampled = smote.fit_resample(X_scaled, y_encoded)

print(f"Original dataset shape: {Counter(y_encoded)}")
print(f"Resampled dataset shape: {Counter(y_resampled)}")


y, sr = librosa.load("seg_sound.wav", sr=None)

#  spectrogram
S = librosa.feature.melspectrogram(y=y, sr=sr)
librosa.display.specshow(librosa.power_to_db(S, ref=np.max), sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram')
plt.show()


X_resampled_reshaped = X_resampled.reshape((X_resampled.shape[0], X_resampled.shape[1], 1))

# إنشاء النموذج
model = Sequential([
    Conv1D(filters=150, kernel_size=5, activation='relu', input_shape=(X_resampled.shape[1], 1)),
    BatchNormalization(),
    Dropout(0.3),
    Conv1D(filters=64, kernel_size=3, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Bidirectional(LSTM(64, return_sequences=False)),
    Dense(64, activation='relu'),
    Dense(len(np.unique(y_resampled)), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 0.003 * 0.97 ** epoch)

# تدريب النموذج
history = model.fit(X_resampled_reshaped, y_resampled, epochs=50, batch_size=32, verbose=1, callbacks=[lr_schedule])

model.save(r'baby_cry_model.h5')
print("Model training is complete.")


#   Training Accuracy curve
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training Accuracy Curve')
plt.show()

# Training Loss Curve
plt.plot(history.history['loss'], label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss Curve')
plt.show()





