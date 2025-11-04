"""
GREEN VOICE: Real-Time Plant Health Monitoring System
Version: 1.0 - Production Ready
Hardware Integration: ESP32 Multi-Sensor Platform
ML Framework: TensorFlow CNN-LSTM Model
Last Updated: November 4, 2025
"""

# ========================================
# LIBRARY IMPORTS
# ========================================
import serial
import json
import time
import numpy as np
import tensorflow as tf
import os
import pandas as pd
import pickle
from datetime import datetime
from collections import Counter

# ========================================
# SYSTEM CONFIGURATION
# ========================================
SERIAL_PORT = 'COM8'
BAUD_RATE = 115200
TIMESTEPS = 16
PREDICTIONS_BEFORE_SUMMARY = 5

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML_DIR = os.path.join(PROJECT_ROOT, 'ML-Model Processing', 'scripts')
MODEL_PATH = os.path.join(ML_DIR, 'models', 'final_cnn_lstm_model.keras')

HARDWARE_FEATURES = [
    "temp_c", "humidity_p", "pressure_hpa",
    "light_lux", "soil_moisture_p", "bio_signal_mv"
]

LOG_DIR = os.path.join(PROJECT_ROOT, 'test_results')
os.makedirs(LOG_DIR, exist_ok=True)

SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
SESSION_LOG = os.path.join(LOG_DIR, f'test_session_{SESSION_ID}.csv')
SESSION_SUMMARY = os.path.join(LOG_DIR, f'summary_{SESSION_ID}.txt')

# ========================================
# MODEL AND ENCODER LOADING
# ========================================
def load_system():
    """Load pre-trained model, feature scaler, and label encoder"""
    print("\n" + "="*70)
    print("GREEN VOICE: PLANT HEALTH MONITORING SYSTEM")
    print("="*70)
    print(f"\nSession: {SESSION_ID}")
    print(f"Project: {PROJECT_ROOT}\n")
    
    # Load CNN-LSTM model
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"[OK] Model loaded: {MODEL_PATH}")
        print(f"     Input: {model.input_shape}")
        print(f"     Output: {model.output_shape}")
        model_features = model.input_shape[2]
        num_classes = model.output_shape[1]
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        exit(1)
    
    # Load MinMax scaler
    scaler = None
    for loc in ['datasets', 'codes']:
        try:
            scaler_path = os.path.join(ML_DIR, loc, 'scaler.pkl')
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            print(f"[OK] Scaler loaded: {scaler_path}")
            break
        except:
            continue
    
    if scaler is None:
        print("[WARNING] Scaler not found, using manual normalization")
    
    # Load label encoder
    label_map = None
    for loc in ['datasets', 'codes']:
        try:
            le_path = os.path.join(ML_DIR, loc, 'label_encoder.pkl')
            with open(le_path, 'rb') as f:
                le = pickle.load(f)
            label_map = {i: str(cls) for i, cls in enumerate(le.classes_)}
            print(f"[OK] Labels loaded: {list(label_map.values())}")
            break
        except:
            continue
    
    if label_map is None:
        print("[WARNING] Using default labels")
        label_map = {0: "Healthy", 1: "Diseased"}
        print(f"      Labels: {list(label_map.values())}")
    
    return model, scaler, label_map, model_features, num_classes

# ========================================
# FEATURE PROCESSING AND NORMALIZATION
# ========================================
def pad_and_normalize(data_buffer, scaler, target_features):
    """Pad 6 hardware features to model input size and normalize"""
    padded_data = []
    for sample in data_buffer:
        padded = np.zeros(target_features, dtype=np.float32)
        padded[:len(sample)] = sample
        padded_data.append(padded)
    
    X = np.array(padded_data).astype('float32')
    
    if scaler:
        X = scaler.transform(X)
    else:
        # Fallback manual normalization with empirical bounds
        mins = np.array([10.0, 30.0, 950.0, 0.0, 0.0, -100.0])
        maxs = np.array([40.0, 95.0, 1050.0, 30000.0, 100.0, 100.0])
        for i in range(len(HARDWARE_FEATURES)):
            X[:, i] = (X[:, i] - mins[i]) / (maxs[i] - mins[i] + 1e-8)
    
    return X.reshape(1, TIMESTEPS, target_features)

# ========================================
# PREDICTION FUNCTIONS
# ========================================
def predict(model, data_buffer, scaler, target_features, label_map):
    """Execute model inference on padded/normalized feature buffer"""
    try:
        X = pad_and_normalize(data_buffer, scaler, target_features)
        predictions = model.predict(X, verbose=0)
        
        pred_idx = np.argmax(predictions[0])
        confidence = predictions[0][pred_idx] * 100
        predicted_class = label_map.get(pred_idx, f"Class_{pred_idx}")
        
        class_probs = {label_map.get(i, f"Class_{i}"): predictions[0][i] * 100
                       for i in range(len(predictions[0]))}
        
        return predicted_class, confidence, class_probs, True
    except Exception as e:
        print(f"[ERROR] Prediction error: {e}")
        return "ERROR", 0.0, {}, False

def determine_health_status(predictions_history):
    """Aggregate multiple predictions to determine final plant health status"""
    if not predictions_history:
        return "UNKNOWN", 0.0
    
    pred_counts = Counter([p['prediction'] for p in predictions_history])
    most_common = pred_counts.most_common(1)[0]
    
    relevant_preds = [p for p in predictions_history 
                     if p['prediction'] == most_common[0]]
    avg_conf = np.mean([p['confidence'] for p in relevant_preds])
    
    return most_common[0], avg_conf

# ========================================
# REPORT GENERATION
# ========================================
def generate_summary(predictions_history, session_start_time):
    """Generate comprehensive test summary with statistics"""
    duration = time.time() - session_start_time
    final_status, avg_conf = determine_health_status(predictions_history)
    
    sensor_data = {
        'temp': [p['sensors']['temp_c'] for p in predictions_history],
        'humidity': [p['sensors']['humidity_p'] for p in predictions_history],
        'soil': [p['sensors']['soil_moisture_p'] for p in predictions_history],
        'bio': [p['sensors']['bio_signal_mv'] for p in predictions_history]
    }
    
    summary = f"""
{'='*70}
GREEN VOICE PLANT HEALTH MONITORING - TEST SUMMARY
{'='*70}

Test Session: {SESSION_ID}
Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)
Total Predictions: {len(predictions_history)}

{'='*70}
FINAL DIAGNOSIS
{'='*70}

PLANT STATUS: {final_status.upper()}
Confidence: {avg_conf:.1f}%

Prediction Breakdown:
"""
    
    pred_counts = Counter([p['prediction'] for p in predictions_history])
    for pred, count in pred_counts.most_common():
        percentage = (count / len(predictions_history)) * 100
        summary += f"  * {pred}: {count} times ({percentage:.1f}%)\n"
    
    summary += f"""
{'='*70}
SENSOR ANALYSIS
{'='*70}

Temperature:
  Average: {np.mean(sensor_data['temp']):.1f} C
  Range: {np.min(sensor_data['temp']):.1f} C - {np.max(sensor_data['temp']):.1f} C

Humidity:
  Average: {np.mean(sensor_data['humidity']):.1f}%
  Range: {np.min(sensor_data['humidity']):.1f}% - {np.max(sensor_data['humidity']):.1f}%

Soil Moisture:
  Average: {np.mean(sensor_data['soil']):.0f}%
  Range: {np.min(sensor_data['soil']):.0f}% - {np.max(sensor_data['soil']):.0f}%

Bio-Signal:
  Average: {np.mean(sensor_data['bio']):.2f} mV
  Range: {np.min(sensor_data['bio']):.2f} mV - {np.max(sensor_data['bio']):.2f} mV

{'='*70}
DATA SAVED
{'='*70}

Detailed Log: {SESSION_LOG}
This Summary: {SESSION_SUMMARY}

{'='*70}
"""
    
    return summary, final_status, avg_conf

# ========================================
# DATA LOGGING
# ========================================
def log_prediction(timestamp, sensors, prediction, confidence, 
                  predictions_history):
    """Persist prediction and sensor data to CSV and session history"""
    log_entry = {
        'timestamp': timestamp,
        'temp_c': sensors['temp_c'],
        'humidity_p': sensors['humidity_p'],
        'pressure_hpa': sensors['pressure_hpa'],
        'light_lux': sensors['light_lux'],
        'soil_moisture_p': sensors['soil_moisture_p'],
        'bio_signal_mv': sensors['bio_signal_mv'],
        'prediction': prediction,
        'confidence': confidence
    }
    
    predictions_history.append({
        'prediction': prediction,
        'confidence': confidence,
        'sensors': sensors.copy()
    })
    
    df = pd.DataFrame([log_entry])
    if not os.path.exists(SESSION_LOG):
        df.to_csv(SESSION_LOG, index=False)
    else:
        df.to_csv(SESSION_LOG, mode='a', header=False, index=False)

# ========================================
# MAIN MONITORING LOOP
# ========================================
def main():
    """Execute real-time monitoring and inference pipeline"""
    model, scaler, label_map, model_features, num_classes = load_system()
    
    print(f"\nConnecting to ESP32 on {SERIAL_PORT}...")
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=5)
        time.sleep(2)
        ser.reset_input_buffer()
        print(f"[OK] Connected!")
    except Exception as e:
        print(f"[ERROR] Connection failed: {e}")
        return
    
    print(f"\nCollecting {TIMESTEPS} samples per prediction...")
    print(f"Summary generated after every {PREDICTIONS_BEFORE_SUMMARY} "
          f"predictions")
    print("="*70 + "\n")
    
    data_buffer = []
    predictions_history = []
    prediction_count = 0
    session_start = time.time()
    
    try:
        while True:
            if ser.in_waiting > 0:
                try:
                    line = ser.readline().decode('utf-8', 
                                                errors='ignore').strip()
                except:
                    continue
                
                if line.startswith('{'):
                    try:
                        data = json.loads(line)
                        sample = [data[key] for key in HARDWARE_FEATURES]
                        data_buffer.append(sample)
                        
                        print(f"Sample {len(data_buffer):2d}/{TIMESTEPS} | "
                              f"T:{data['temp_c']:5.1f}C "
                              f"H:{data['humidity_p']:5.1f}% "
                              f"S:{data['soil_moisture_p']:3.0f}% "
                              f"Bio:{data['bio_signal_mv']:7.2f}mV")
                        
                        if len(data_buffer) >= TIMESTEPS:
                            prediction_count += 1
                            
                            pred_class, conf, probs, success = predict(
                                model, data_buffer, scaler, 
                                model_features, label_map
                            )
                            
                            if success:
                                status_marker = ("[HEALTHY]" if "health" 
                                               in pred_class.lower() 
                                               else "[DISEASED]")
                                print(f"\n{status_marker} PREDICTION "
                                      f"#{prediction_count}: {pred_class} "
                                      f"({conf:.1f}%)")
                                
                                log_prediction(data['timestamp'], data, 
                                             pred_class, conf, 
                                             predictions_history)
                                
                                if (prediction_count % 
                                    PREDICTIONS_BEFORE_SUMMARY == 0):
                                    summary, status, avg_conf = (
                                        generate_summary(
                                            predictions_history, 
                                            session_start
                                        ))
                                    print("\n" + summary)
                                    
                                    with open(SESSION_SUMMARY, 'w', 
                                            encoding='utf-8') as f:
                                        f.write(summary)
                                    
                                    print(f"[OK] Summary saved to: "
                                          f"{SESSION_SUMMARY}\n")
                            
                            data_buffer.pop(0)
                    
                    except (json.JSONDecodeError, KeyError):
                        pass
            
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print(f"\n\nTest stopped by user")
        
        if predictions_history:
            summary, status, avg_conf = generate_summary(
                predictions_history, session_start
            )
            print("\n" + summary)
            
            with open(SESSION_SUMMARY, 'w', encoding='utf-8') as f:
                f.write(summary)
            
            print(f"\n[OK] All data saved to: {LOG_DIR}")
    
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
        print("\nGreen Voice monitoring complete!")

if __name__ == "__main__":
    main()
