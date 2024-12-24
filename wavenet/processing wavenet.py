import os
import librosa
import soundfile as sf

main_folder_path = r"clean baby sounds"
output_folder = r"processed_sounds"
os.makedirs(output_folder, exist_ok=True)
print(f"Processing files in folder: {main_folder_path}")

for root, dirs, files in os.walk(main_folder_path):
    for file_name in files:
        file_path = os.path.join(root, file_name)

        if file_name.endswith(('.wav', '.mp3', '.ogg')):
            try:
                print(f"Processing: {file_path}")
                audio, sr = librosa.load(file_path, sr=16000)
                audio = librosa.util.fix_length(audio, size=16000 * 3)  # 3 ثوانٍ
                
                relative_path = os.path.relpath(root, main_folder_path)
                output_subfolder = os.path.join(output_folder, relative_path)
                os.makedirs(output_subfolder, exist_ok=True)
                
                output_path = os.path.join(output_subfolder, f"{os.path.splitext(file_name)[0]}_processed.wav")
                sf.write(output_path, audio, 16000)
                print(f"Saved: {output_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

print("Processing complete.")





