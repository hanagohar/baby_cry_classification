import os
import librosa
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

train_folder = r'clean baby sounds'  
test_folder = r'clean baby sound'  
output_t_file = r'test_data.csv'  
output_tt_file = r'train_data.csv'  
train_features = []
train_labels = []
test_features = []  
print("Processing classified data...")
for root, dirs, files in os.walk(train_folder):
    for file in files:
        if file.endswith(('.wav', '.ogg')):
            file_path = os.path.join(root, file)
            label = os.path.basename(root)  
            y, sr = librosa.load(file_path, sr=None)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=30)
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y).mean()
            rms = librosa.feature.rms(y=y).mean()
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr).mean()
            mfccs_mean = np.mean(mfccs, axis=1)  # متوسط MFCCs
            mfccs_std = np.std(mfccs, axis=1)    # انحراف معياري
            feature_row = list(mfccs_mean) + list(mfccs_std) + [
                spectral_centroid, spectral_bandwidth, spectral_rolloff,
                zero_crossing_rate, rms, chroma_stft ]
            train_features.append(feature_row)
            train_labels.append(label)
encoder = LabelEncoder()
train_labels_encoded = encoder.fit_transform(train_labels)
X_train, X_test, y_train, y_test = train_test_split(
    train_features, train_labels_encoded, test_size=0.2, random_state=42, stratify=train_labels_encoded)
print("Training the model...")
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')  # إضافة class_weight
model.fit(X_train, y_train)
# تصنيف البيانات  
print("Classifying unclassified data...")
for root, dirs, files in os.walk(test_folder):
    for file in files:
        if file.endswith(('.wav', '.mp3')):
            file_path = os.path.join(root, file)
            y, sr = librosa.load(file_path, sr=None)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=30)
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y).mean()
            rms = librosa.feature.rms(y=y).mean()
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr).mean()
            mfccs_mean = np.mean(mfccs, axis=1)  
            mfccs_std = np.std(mfccs, axis=1)    
            feature_row = list(mfccs_mean) + list(mfccs_std) + [
                spectral_centroid, spectral_bandwidth, spectral_rolloff,
                zero_crossing_rate, rms, chroma_stft ]
            test_features.append(feature_row)
test_predictions = model.predict(test_features)
test_labels = encoder.inverse_transform(test_predictions)
test_labels_encoded = test_predictions  
columns = [f'MFCC_mean_{i+1}' for i in range(30)] + \
          [f'MFCC_std_{i+1}' for i in range(30)] + \
          ['Spectral_Centroid', 'Spectral_Bandwidth', 'Spectral_Rolloff',
           'Zero_Crossing_Rate', 'RMS_Energy', 'Chroma_STFT']
df_test = pd.DataFrame(test_features, columns=columns)
df_test['label'] = test_labels_encoded  
df_test.to_csv(output_tt_file, index=False)
df_train = pd.DataFrame(X_train, columns=columns)
df_train['label'] = y_train  
df_train.to_csv(output_t_file, index=False)
print(f"Classification completed. Results saved to {output_tt_file}.")
print(f"Training data saved to {output_t_file}.")



