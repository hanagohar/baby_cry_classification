import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

model = tf.keras.models.load_model(r'baby_cry_model.h5')
scaler = joblib.load(r'scaler.pkl')
encoder = joblib.load(r'label_encoder.pkl')
encoder.classes_ = np.array(["belly_pain", "burping", "discomfort", "hungry", "tired"])
new_data_file = r'predic_data.csv'
new_data = pd.read_csv(new_data_file)
X_new = new_data.values
expected_num_features = 66
if X_new.shape[1] != expected_num_features:
    raise ValueError(f"Number of features in new data ({X_new.shape[1]}) does not match the expected number of features ({expected_num_features})")
X_new_scaled = scaler.transform(X_new)
X_new_scaled_reshaped = X_new_scaled.reshape((X_new_scaled.shape[0], X_new_scaled.shape[1], 1))
y_pred = model.predict(X_new_scaled_reshaped)
y_pred_classes = np.argmax(y_pred, axis=1)
print("Predicted Classes (Numeric) for New Data:")
print(y_pred_classes)  
#Histogram of Predictions
plt.figure(figsize=(10, 6))
plt.hist(y_pred_classes, bins=np.arange(len(encoder.classes_) + 1) - 0.5, rwidth=0.8)
plt.title('Distribution of Predicted Classes')
plt.xlabel('Predicted Class')
plt.ylabel('Frequency')
plt.xticks(np.arange(len(encoder.classes_)), encoder.classes_, rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()






