import serial
import json
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import os

# --- 1. CONFIGURATION ---
# >>> CHANGE THIS TO YOUR ESP32's SERIAL PORT <<<
SERIAL_PORT = '/dev/ttyUSB0'  # Example: Use 'COM3' on Windows
BAUD_RATE = 115200             # Must match your Arduino code
# Time steps for the CNN-LSTM model. Your model was trained with sequences of this length.
# This means we need to buffer this many samples before making a prediction.
TIMESTEPS = 16 

models_dir = '../models'
datasets_dir = '../datasets'

# --- 2. LOAD MODEL AND UTILITIES ---
print("Loading model and utilities...")
try:
    model_path = os.path.join(models_dir, 'final_cnn_lstm_model.keras')
    model = tf.keras.models.load_model(model_path)
    with open(os.path.join(datasets_dir, 'label_encoder.pkl'), 'rb') as f:
        label_encoder = pickle.load(f)
    with open(os.path.join(datasets_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    print("Model and utilities loaded successfully.")
except Exception as e:
    print(f"Error loading model/utilities. Check file paths: {e}")
    exit()

# --- 3. HELPER FUNCTIONS (Adapted from your predict code) ---

def generate_status_text(disease_class, confidence):
    """Generates user-friendly status text based on prediction and confidence."""
    dcstr = str(disease_class).lower()
    
    # Check for keywords and set confidence thresholds
    if 'healthy' in dcstr or 'normal' in dcstr:
        if confidence >= 85:
            return f"🌿 HEALTHY ({confidence:.1f}%)"
        else:
            return f"🌱 MAYBE HEALTHY ({confidence:.1f}%)"
    elif 'disease' in dcstr or 'blight' in dcstr or 'rust' in dcstr:
        if confidence >= 85:
            return f"🚨 {disease_class} ({confidence:.1f}%)"
        else:
            return f"⚠ POTENTIAL {disease_class} ({confidence:.1f}%)"
    else:
        return f"❓ {disease_class} ({confidence:.1f}%) - Review"

# --- NEW: Function to process and predict a single sample buffer ---
def predict_from_buffer(data_buffer):
    """
    Takes a 2D NumPy array (TIMESTEPS x features) and makes a prediction.
    """
    # X is already a buffer of TIMESTEPS samples
    X = np.array(data_buffer).astype('float32')

    # 1. Scale the data (Scaler expects (n_samples, n_features))
    # We treat the entire buffer (TIMESTEPS rows) as n_samples
    X_scaled = scaler.transform(X)

    # 2. Reshape for CNN-LSTM: (1, TIMESTEPS, n_features)
    # The reshape_for_lstm from your original code is designed for a single CSV row.
    # For a real-time buffer, we reshape directly:
    n_features = X_scaled.shape[1]
    X_lstm = X_scaled.reshape(1, TIMESTEPS, n_features)

    # 3. Predict
    predictions = model.predict(X_lstm, verbose=0)
    
    # 4. Process results
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    predicted_confidence = np.max(predictions, axis=1)[0] * 100
    predicted_label = label_encoder.inverse_transform([predicted_class_idx])[0]
    
    status_text = generate_status_text(predicted_label, predicted_confidence)

    return predicted_label, predicted_confidence, status_text

# --- 4. SERIAL COMMUNICATION AND MAIN LOOP ---

# List to store the features for the TIMESTEPS buffer
# The order of features must match the order the model was trained on!
feature_keys = [
    "temp_c", "humidity_p", "pressure_hpa", 
    "light_lux", "soil_moisture_p", "bio_signal_mv"
]

data_buffer = []

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=5)
    time.sleep(2) # Give time for the connection to establish
    print(f"\nConnected to ESP32 on {SERIAL_PORT}. Waiting for {TIMESTEPS} samples...")
    print("-" * 50)
    
    # Clear any leftover data in the serial buffer
    ser.reset_input_buffer() 

    while True:
        # Read the entire line (JSON object) from the ESP32
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()
            
            if line.startswith('{'): 
                try:
                    # Parse the JSON string
                    data = json.loads(line)
                    
                    # Extract features in the correct order
                    new_sample = [data[key] for key in feature_keys]
                    
                    # Add the new sample to the buffer
                    data_buffer.append(new_sample)

                    print(f"Sample received ({len(data_buffer)}/{TIMESTEPS}): {new_sample}")
                    
                    # Check if we have enough data points (TIMESTEPS) to make a prediction
                    if len(data_buffer) >= TIMESTEPS:
                        
                        # --- Make Prediction ---
                        pred_label, pred_conf, status = predict_from_buffer(data_buffer)
                        
                        print("\n" + "=" * 50)
                        print("✨ NEW PLANT CONDITION PREDICTION ✨")
                        print(f"Predicted Class: *{pred_label}*")
                        print(f"Confidence: *{pred_conf:.2f}%*")
                        print(f"Status: *{status}*")
                        print("=" * 50 + "\n")
                        
                        # --- IMPORTANT: Shift the buffer ---
                        # For continuous monitoring, we remove the oldest sample 
                        # to make room for the next, maintaining a sliding window of TIMESTEPS.
                        data_buffer.pop(0)

                except json.JSONDecodeError:
                    # This happens if the line is incomplete or corrupted
                    print(f"Skipping incomplete line: {line[:50]}...")
                except Exception as e:
                    print(f"An unexpected error occurred during prediction: {e}")
                    
        time.sleep(0.1) # Small delay to prevent burning up CPU

except serial.SerialException as e:
    print(f"\nCRITICAL: Failed to connect to serial port {SERIAL_PORT}. {e}")
    print("Please check the port name, baud rate, and connection.")
except KeyboardInterrupt:
    print("\nPrediction loop stopped by user.")
finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()
        print("Serial port closed.")
