import serial
import json
import time
import numpy as np
import tensorflow as tf
import os
import pandas as pd
import pickle

# ===============================================
# === 1. CONFIGURATION ===
# ===============================================

SERIAL_PORT = 'COM8'
BAUD_RATE = 115200

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML_PROCESS_DIR = os.path.join(PROJECT_ROOT, 'ML-Model Processing')
MODELS_DIR = os.path.join(ML_PROCESS_DIR, 'scripts', 'models')
CODES_DIR = os.path.join(ML_PROCESS_DIR, 'scripts', 'codes')
MODEL_PATH = os.path.join(MODELS_DIR, 'final_cnn_lstm_model.keras')
LABEL_ENCODER_PATH = os.path.join(CODES_DIR, 'label_encoder.pkl')  # Try codes dir

# Hardware provides 6 features
HARDWARE_FEATURE_KEYS = [
    "temp_c", "humidity_p", "pressure_hpa", 
    "light_lux", "soil_moisture_p", "bio_signal_mv"
]

EXPECTED_FEATURES = 82
TIMESTEPS = 16
DEFAULT_FEATURE_VALUE = 0.0

LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, 'plant_health_log.csv')

# ===============================================
# === 2. LOAD MODEL & LABEL ENCODER ===
# ===============================================

print("\n" + "="*60)
print("🌱 GREEN VOICE: PLANT HEALTH MONITORING SYSTEM")
print("="*60)
print(f"\n📂 Project Root: {PROJECT_ROOT}")
print(f"📂 Model Path: {MODEL_PATH}")
print(f"🔌 Serial Port: {SERIAL_PORT}\n")

# Load model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
    print(f"   Input shape: {model.input_shape}")
    print(f"   Output shape: {model.output_shape}")
    
    model_timesteps = model.input_shape[1]
    model_features = model.input_shape[2]
    num_classes = model.output_shape[1]
    
    print(f"\n📊 Model Configuration:")
    print(f"   Timesteps: {model_timesteps}")
    print(f"   Features: {model_features}")
    print(f"   Classes: {num_classes}")
    print(f"\n🔧 Hardware: {len(HARDWARE_FEATURE_KEYS)} features → Padding to {model_features}")
    
except Exception as e:
    print(f"❌ ERROR loading model: {e}")
    exit(1)

# Load label encoder (try multiple locations)
label_encoder_loaded = False
for path in [LABEL_ENCODER_PATH, 
             os.path.join(ML_PROCESS_DIR, 'scripts', 'datasets', 'label_encoder.pkl'),
             os.path.join(CODES_DIR, '../datasets/label_encoder.pkl')]:
    try:
        with open(path, 'rb') as f:
            label_encoder = pickle.load(f)
        LABEL_MAP = {i: label for i, label in enumerate(label_encoder.classes_)}
        print(f"\n✅ Label encoder loaded from: {path}")
        print(f"   Classes: {list(LABEL_MAP.values())}")
        label_encoder_loaded = True
        break
    except:
        continue

if not label_encoder_loaded:
    print(f"\n⚠️  Label encoder not found, using default labels")
    # Create default labels based on number of classes
    if num_classes == 2:
        LABEL_MAP = {0: "Healthy", 1: "Diseased"}
    elif num_classes == 3:
        LABEL_MAP = {0: "Healthy", 1: "Early Disease", 2: "Severe Disease"}
    elif num_classes == 4:
        LABEL_MAP = {0: "Healthy", 1: "Early Disease", 2: "Drought Stress", 3: "Pest Damage"}
    else:
        LABEL_MAP = {i: f"Class_{i}" for i in range(num_classes)}
    print(f"   Default classes: {list(LABEL_MAP.values())}")

# ===============================================
# === 3. FEATURE PROCESSING ===
# ===============================================

def pad_features_to_82(hardware_features_6):
    """Pad 6 hardware features to 82 features"""
    padded_features = list(hardware_features_6)
    padding_needed = EXPECTED_FEATURES - len(HARDWARE_FEATURE_KEYS)
    padded_features.extend([DEFAULT_FEATURE_VALUE] * padding_needed)
    return np.array(padded_features)

def prepare_sequence_for_model(data_buffer):
    """Convert hardware data buffer to model input"""
    padded_sequence = []
    for hardware_sample in data_buffer:
        padded_sample = pad_features_to_82(hardware_sample)
        padded_sequence.append(padded_sample)
    
    X = np.array(padded_sequence).astype('float32')
    X_scaled = normalize_features(X)
    X_model = X_scaled.reshape(1, TIMESTEPS, EXPECTED_FEATURES)
    
    return X_model

def normalize_features(X):
    """Normalize features"""
    X_norm = X.copy()
    
    # Normalize first 6 hardware features
    feature_mins = np.array([10.0, 30.0, 950.0, 0.0, 0.0, -100.0])
    feature_maxs = np.array([40.0, 95.0, 1050.0, 30000.0, 100.0, 100.0])
    
    for i in range(len(HARDWARE_FEATURE_KEYS)):
        X_norm[:, i] = (X[:, i] - feature_mins[i]) / (feature_maxs[i] - feature_mins[i] + 1e-8)
    
    return X_norm

# ===============================================
# === 4. PREDICTION FUNCTIONS ===
# ===============================================

def get_label_from_index(index, confidence):
    """Translate index to label"""
    label = LABEL_MAP.get(index, "UNKNOWN")
    
    if confidence < 60:
        return f"❓ UNCERTAIN ({confidence:.1f}%)", "🟡"
    elif any(word in label.lower() for word in ['healthy', 'normal', 'good']):
        return f"🌿 {label} ({confidence:.1f}%)", "🟢"
    else:
        return f"🚨 {label} ({confidence:.1f}%)", "🔴"

def predict_from_buffer(data_buffer):
    """Make prediction"""
    try:
        X_model = prepare_sequence_for_model(data_buffer)
        predictions = model.predict(X_model, verbose=0)
        
        predicted_class_idx = np.argmax(predictions, axis=1)[0]
        predicted_confidence = np.max(predictions, axis=1)[0] * 100
        
        class_probs = {LABEL_MAP.get(i, f"Class_{i}"): predictions[0][i] * 100 
                       for i in range(len(predictions[0]))}
        
        status_text, emoji = get_label_from_index(predicted_class_idx, predicted_confidence)
        
        return status_text, emoji, predicted_confidence, class_probs, True
        
    except Exception as e:
        print(f"\n❌ Prediction error: {e}")
        return "ERROR", "❌", 0.0, {}, False

def log_prediction(timestamp, sensor_data, prediction, confidence):
    """Log to CSV"""
    try:
        log_data = {
            'timestamp': timestamp,
            'temp_c': sensor_data['temp_c'],
            'humidity_p': sensor_data['humidity_p'],
            'pressure_hpa': sensor_data['pressure_hpa'],
            'light_lux': sensor_data['light_lux'],
            'soil_moisture_p': sensor_data['soil_moisture_p'],
            'bio_signal_mv': sensor_data['bio_signal_mv'],
            'prediction': prediction,
            'confidence': confidence
        }
        
        df = pd.DataFrame([log_data])
        
        if not os.path.exists(LOG_FILE):
            df.to_csv(LOG_FILE, index=False)
        else:
            df.to_csv(LOG_FILE, mode='a', header=False, index=False)
            
    except Exception as e:
        pass  # Silent fail for logging

# ===============================================
# === 5. MAIN LOOP WITH ERROR HANDLING ===
# ===============================================

data_buffer = []
prediction_count = 0
error_count = 0

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=5)
    time.sleep(2)
    ser.reset_input_buffer()
    
    print(f"\n✅ Connected to ESP32 successfully!")
    print(f"⏳ Collecting {TIMESTEPS} samples (approximately {TIMESTEPS * 5} seconds)...")
    print("="*60 + "\n")
    
    while True:
        try:
            if ser.in_waiting > 0:
                # Read with error handling
                try:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                except UnicodeDecodeError:
                    error_count += 1
                    if error_count % 10 == 0:
                        print(f"⚠️  {error_count} decode errors (ignoring corrupt data)")
                    continue
                
                # Parse JSON
                if line.startswith('{'):
                    try:
                        data = json.loads(line)
                        
                        # Extract features
                        new_sample = [data[key] for key in HARDWARE_FEATURE_KEYS]
                        data_buffer.append(new_sample)
                        
                        # Display progress
                        print(f"📊 Sample {len(data_buffer):2d}/{TIMESTEPS} | "
                              f"Temp: {data['temp_c']:5.1f}°C | "
                              f"Humid: {data['humidity_p']:5.1f}% | "
                              f"Soil: {data['soil_moisture_p']:3.0f}% | "
                              f"Bio: {data['bio_signal_mv']:7.2f}mV")
                        
                        # Predict when buffer full
                        if len(data_buffer) >= TIMESTEPS:
                            prediction_count += 1
                            
                            status, emoji, confidence, class_probs, success = predict_from_buffer(data_buffer)
                            
                            if success:
                                # Display results
                                print("\n" + "="*60)
                                print(f"{emoji} PREDICTION #{prediction_count}")
                                print("="*60)
                                print(f"🔮 Status: {status}\n")
                                print(f"📈 Class Probabilities:")
                                for class_name, prob in sorted(class_probs.items(), 
                                                               key=lambda x: x[1], 
                                                               reverse=True):
                                    bar = "█" * int(prob / 2)
                                    print(f"   {class_name:20s}: {bar:25s} {prob:5.1f}%")
                                
                                print(f"\n📊 Sensor Readings:")
                                print(f"   Temperature:    {data['temp_c']:5.1f}°C")
                                print(f"   Humidity:       {data['humidity_p']:5.1f}%")
                                print(f"   Pressure:       {data['pressure_hpa']:7.2f} hPa")
                                print(f"   Soil Moisture:  {data['soil_moisture_p']:3.0f}%")
                                print(f"   Bio-Signal:     {data['bio_signal_mv']:7.2f} mV")
                                print(f"   Light:          {data['light_lux']:6.0f} lux")
                                print("="*60 + "\n")
                                
                                log_prediction(data['timestamp'], data, status, confidence)
                            
                            # Rolling window
                            data_buffer.pop(0)
                            
                    except json.JSONDecodeError:
                        pass
                    except KeyError as e:
                        print(f"⚠️  Missing key: {e}")
                    except Exception as e:
                        print(f"⚠️  Error: {e}")
            
            time.sleep(0.01)  # Small delay
            
        except serial.SerialException as e:
            print(f"\n❌ Serial error: {e}")
            print("Attempting to reconnect...")
            time.sleep(2)
            try:
                ser.close()
                ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=5)
                print("✅ Reconnected!")
            except:
                break

except KeyboardInterrupt:
    print(f"\n\n⏹️  Stopped by user")
    print(f"📊 Predictions: {prediction_count} | Errors: {error_count}")
    print(f"💾 Log: {LOG_FILE}")

except Exception as e:
    print(f"\n❌ Fatal error: {e}")

finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()
        print("🔌 Serial closed")
    print("\n👋 Thank you for using Green Voice!")
