import tkinter as tk
import tkinter.font as tkFont
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import pyaudio
import wave
import tensorflow as tf
import numpy as np
import librosa
import joblib

model = tf.keras.models.load_model(r'baby_cry_model.h5')
encoder = joblib.load(r'label_encoder.pkl')
scaler = joblib.load(r'scaler.pkl')
class RoundedButton(tk.Canvas):
    def __init__(self, master, text, command, color="pink", hover_color="lightpink", **kwargs):
        super().__init__(master, **kwargs)
        self.command = command
        self.text = text
        self.default_color = color
        self.hover_color = hover_color
        self.draw_button()
        self.bind("<Button-1>", self.on_click)
        self.bind("<Enter>", self.on_hover)
        self.bind("<Leave>", self.on_leave)
    def draw_button(self):
        self.create_oval(5, 5, 195, 75, fill=self.default_color, outline="")
        self.create_text(100, 40, text=self.text, fill="black", font=("Arial", 16))
    def on_click(self, event):
        self.command()
    def on_hover(self, event):
        self.itemconfig(1, fill=self.hover_color)
    def on_leave(self, event):
        self.itemconfig(1, fill=self.default_color)
class MamaApp:
    def __init__(self, master):
        self.master = master
        self.master.title("I’m just a mama")
        self.master.geometry("750x800")
        self.audio_file = None
        self.setup_welcome_screen()
    def setup_welcome_screen(self):
        self.clear_widgets()
        self.welcome_image = Image.open("mom and baby.jpg")
        self.welcome_image = self.welcome_image.resize((400, 300))
        self.photo = ImageTk.PhotoImage(self.welcome_image)
        self.image_label = tk.Label(self.master, image=self.photo)
        self.image_label.pack(pady=20)
        self.title_label = tk.Label(self.master, text="I’m just a mama", font=("Comic Sans MS", 28), 
                                    fg="#D76C82")
        self.title_label.pack(pady=20)
        self.welcome_message = tk.Label(self.master, text="Welcome, dear mom!\nThis app helps you understand why your baby is crying.\nSimply record or upload an audio file and let us help you.",
                                         font=tkFont.Font(family="Times New Roman", size=18), fg="#2A3335", justify="center")
        self.welcome_message.pack(pady=20)
        self.start_button = RoundedButton(self.master, text="Start", command=self.setup_main_interface, width=200, height=70)
        self.start_button.pack(pady=30)
    def setup_main_interface(self):
        self.clear_widgets()
        self.audio_image = Image.open("cry baby.jpg")
        self.audio_image = self.audio_image.resize((400, 300))
        self.audio_photo = ImageTk.PhotoImage(self.audio_image)
        self.audio_image_label = tk.Label(self.master, image=self.audio_photo)
        self.audio_image_label.pack(pady=10)
        self.label = tk.Label(self.master, text="Record your child's voice or choose a file", font=("Comic Sans MS", 24), fg="#D76C82")
        self.label.pack(pady=10)
        button_frame = tk.Frame(self.master)
        button_frame.pack(pady=10)
        self.record_button = RoundedButton(button_frame, text="Record New", command=self.record_audio, width=200, height=70, color="#FFCCCB", hover_color="#FFB6C1")
        self.record_button.grid(row=0, column=0, padx=10, pady=5)
        self.upload_button = RoundedButton(button_frame, text="Upload Audio File", command=self.upload_audio, width=200, height=70, color="#FFCCCB", hover_color="#FFB6C1")
        self.upload_button.grid(row=0, column=1, padx=10, pady=5)
        self.analyze_button = RoundedButton(self.master, text="Analyze Audio", command=self.analyze_audio, width=200, height=70, color="#FFCCCB", hover_color="#FFB6C1")
        self.analyze_button.pack(pady=10)
        self.back_button = RoundedButton(self.master, text="Back", command=self.setup_welcome_screen, width=200, height=70, color="#C4E1F6", hover_color="#A2D2DF")
        self.back_button.pack(pady=10)
    def clear_widgets(self):
        for widget in self.master.winfo_children():
            widget.destroy()
    def record_audio(self):
        chunk = 1024
        format = pyaudio.paInt16
        channels = 1
        rate = 44100
        seconds = 5
        filename = "child_voice.wav"
        p = pyaudio.PyAudio()
        stream = p.open(format=format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)
        frames = []
        messagebox.showinfo("Recording", "Start recording...")
        for i in range(0, int(rate / chunk * seconds)):
            data = stream.read(chunk)
            frames.append(data)
        messagebox.showinfo("Recording", "Recording finished.")
        stream.stop_stream()
        stream.close()
        p.terminate()
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(p.get_sample_size(format))
            wf.setframerate(rate)
            wf.writeframes(b''.join(frames))
        self.audio_file = filename
    def upload_audio(self):
        self.audio_file = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
        if self.audio_file:
            messagebox.showinfo("File Uploaded", f"File uploaded: {self.audio_file}")
    def analyze_audio(self):
        if not self.audio_file:
            messagebox.showwarning("Analyze Audio", "Please record or upload an audio file first.")
            return
        try:
            y, sr = librosa.load(self.audio_file, sr=None)
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
                zero_crossing_rate, rms, chroma_stft
            ]
            features_scaled = scaler.transform([feature_row])
            prediction = model.predict(features_scaled)
            predicted_label = np.argmax(prediction)
            predicted_class = encoder.inverse_transform([predicted_label])[0]
            if predicted_class == 3 :
                advice = "The baby seems hungry. Try feeding them. Ensure a quiet and calm environment while feeding."
            elif predicted_class == 4 :
                advice = "The baby seems tired. Rock the baby gently and create a quiet, dimly lit environment for them to sleep."
            elif predicted_class == 0 :
                advice = "The baby seems to have belly pain. Try massaging their belly gently in a circular motion."
            elif predicted_class == 2 :
                advice = "The baby seems uncomfortable. Check their diaper, position, or clothing for anything causing irritation."
            elif predicted_class == 1 :
                advice = "The baby needs to burp. Hold the baby upright and pat their back gently."

            #messagebox.showinfo("Prediction", f"The baby is likely to be: {predicted_class}")
            messagebox.showinfo("Advice", advice)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while analyzing the audio: {str(e)}")
if __name__ == "__main__":
    root = tk.Tk()
    app = MamaApp(root)
    root.mainloop()



