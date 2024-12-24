import os   
import librosa     
import noisereduce as nr    
import soundfile as sf   

input_folder = r"baby sound" 
output_folder = r"clean baby sound"  
os.makedirs(output_folder, exist_ok=True)
def process_audio_files(input_dir, output_dir):    
    for root, dirs, files in os.walk(input_dir):  
        for file in files:
            if file.endswith((".wav")):  
                input_file = os.path.join(root, file)   
                relative_path = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(output_dir, relative_path)
                os.makedirs(output_subdir, exist_ok=True)
                output_file = os.path.join(output_subdir, file)   
                print(f"Processing: {input_file} -> {output_file}") 
                try:
                    y, sr = librosa.load(input_file, sr=None)
                    noise_part = y[:int(sr * 1)] 
                    y_cleaned = nr.reduce_noise(y=y, sr=sr, y_noise=noise_part)
                    sf.write(output_file, y_cleaned, sr)
                    print(f"File saved: {output_file}")
                except Exception as e:
                    print(f"Error processing {input_file}: {e}")
process_audio_files(input_folder, output_folder)
print("Processing complete!")








