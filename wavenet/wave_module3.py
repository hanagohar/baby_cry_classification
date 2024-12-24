import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
import soundfile as sf  # مكتبة لحفظ الصوت بصيغة WAV

def load_audio_files(folder_path, target_sr=16000, duration=3):
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.wav'):
                file_paths.append(os.path.join(root, file))
    if len(file_paths) == 0:
        print("There are no audio files in the specified folder.")
        return np.array([])
    audio_data = []
    for file_path in file_paths:
        try:
            audio, sr = librosa.load(file_path, sr=target_sr, duration=duration)
            audio = librosa.util.fix_length(audio, size=target_sr * duration)
            audio_data.append(audio)
        except Exception as e:
            print(f" error in load  {file_path}: {e}")
    return np.array(audio_data)
audio_folder = r"train"
X = load_audio_files(audio_folder)
if X.size == 0:
    print("Failed to load audio data.")
else:
    print(f" {X.shape[0]} Audio file successfully uploaded. ")
X = X.reshape(-1, 16000, 1)
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
# بناء نموذج WaveNet
def build_wavenet(input_shape):
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Conv1D(64, kernel_size=2, dilation_rate=1, padding='causal', activation='relu'))
    model.add(layers.Conv1D(128, kernel_size=2, dilation_rate=2, padding='causal', activation='relu'))
    model.add(layers.Conv1D(256, kernel_size=2, dilation_rate=4, padding='causal', activation='relu'))
    model.add(layers.Conv1D(1, kernel_size=1, padding='same', activation='tanh'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
input_shape = (16000, 1)
model = build_wavenet(input_shape)
history = model.fit(X_train, X_train, epochs=50, batch_size=32, validation_data=(X_test, X_test))
model.save('wavenet_model.h5')
def generate_and_save_audio(model, test_data, num_files, save_folder):
    os.makedirs(save_folder, exist_ok=True)  
    for i in range(num_files):
        input_audio = test_data[i].reshape(1, 16000, 1)  
        generated_audio = model.predict(input_audio).flatten()  
        output_file = os.path.join(save_folder, f"generated_audio_{i+1}.wav")
        sf.write(output_file, generated_audio, 16000)  
        print(f"File saved:{output_file}")
num_files_to_generate = 50
output_folder = r"generated_audio"
generate_and_save_audio(model, X_test, num_files_to_generate, output_folder)
















