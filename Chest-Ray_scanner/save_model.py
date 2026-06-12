import tensorflow as tf
from tensorflow import keras
import os

# Load the saved h5 file
loaded_model = tf.keras.models.load_model(
    r"C:\Users\el_bostan\Desktop\مشروع FCAI\X_Ray Image\chest_xray\model\chest_xray_model.h5",
    compile=False
)

# Compile it
loaded_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Save in .keras format
os.makedirs(r"C:\Users\el_bostan\Desktop\مشروع FCAI\X_Ray Image\chest_xray\model", exist_ok=True)
loaded_model.save(r"C:\Users\el_bostan\Desktop\مشروع FCAI\X_Ray Image\chest_xray\model\chest_xray_model.keras")

print("✅ Model saved successfully!")