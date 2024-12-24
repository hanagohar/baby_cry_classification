import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

model = tf.keras.models.load_model(r'baby_cry_model.h5')
scaler = joblib.load(r'scaler.pkl')
encoder = joblib.load(r'label_encoder.pkl')
test_data_file = r'test_data.csv'
test_data = pd.read_csv(test_data_file)
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values
X_test_scaled = scaler.transform(X_test)
X_test_scaled_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))
y_pred = model.predict(X_test_scaled_reshaped)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_encoded = encoder.transform(y_test)
accuracy = accuracy_score(y_test_encoded, y_pred_classes)
label_map = {0: "belly_pain", 1: "burping", 2: "discomfort", 3: "hungry", 4: "tired"}
target_names = [label_map[label] for label in encoder.classes_]
report = classification_report(y_test_encoded, y_pred_classes, target_names=target_names)
print(f"Accuracy on test data: {accuracy:.2f}")
print("Classification Report:")
print(report)
# Confusion Matrix
cm = confusion_matrix(y_test_encoded, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.classes_)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()




