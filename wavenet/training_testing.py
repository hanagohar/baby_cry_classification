import os
import shutil
from sklearn.model_selection import train_test_split

root_folder = r'processed_sounds'
train_folder = r'train'
test_folder = r'test'

os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

all_files = []
for root, dirs, files in os.walk(root_folder):
    for file in files:
        if file.endswith(('.wav', '.mp3', '.ogg')): 
            all_files.append(os.path.join(root, file))

train_files, test_files = train_test_split(all_files, test_size=0.2, random_state=42)

for file in train_files:
    destination = os.path.join(train_folder, os.path.relpath(file, root_folder))
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    shutil.copy(file, destination)

for file in test_files:
    destination = os.path.join(test_folder, os.path.relpath(file, root_folder))
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    shutil.copy(file, destination)

print("The files were successfully split into training and test sets.")




