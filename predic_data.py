import os
import librosa
import numpy as np
import pandas as pd

folder_path = "clean_cry"
features_list = []
for file_name in os.listdir(folder_path):
    if file_name.endswith('.wav') or file_name.endswith('.ogg'):
        file_path = os.path.join(folder_path, file_name)
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
        features = list(mfccs_mean) + list(mfccs_std) + [
            spectral_centroid, spectral_bandwidth, spectral_rolloff,
            zero_crossing_rate, rms, chroma_stft ]
        features_list.append(features)
df = pd.DataFrame(features_list)
output_file = 'predic_data.csv'
df.to_csv(output_file, index=False)
print(f"Features have been extracted and saved to {output_file}")




