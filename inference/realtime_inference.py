"""
Real-Time Plant Health Inference Pipeline
Bridges Arduino sensor telemetry to trained CNN-LSTM classification model.
"""

import numpy as np
import tensorflow as tf
import serial
import json
import pickle
import os
from collections import deque
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PlantHealthInferenceEngine:
    
    def __init__(
        self,
        model_path: str,
        label_encoder_path: str,
        serial_port: str,
        baud_rate: int = 115200,
        window_size: int = 10
    ):
        self.model_path = model_path
        self.label_encoder_path = label_encoder_path
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.window_size = window_size
        
        self.model = None
        self.label_encoder = None
        self.serial_conn = None
        self.feature_buffer = deque(maxlen=window_size)
        
    def initialize(self) -> None:
        logger.info("Initializing inference engine...")
        self._load_model()
        self._load_encoder()
        self._establish_serial_connection()
        logger.info("Initialization complete.")
        
    def _load_model(self) -> None:
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        self.model = tf.keras.models.load_model(self.model_path)
        logger.info(f"Model loaded: {self.model_path}")
        
    def _load_encoder(self) -> None:
        if not os.path.exists(self.label_encoder_path):
            raise FileNotFoundError(f"Encoder not found: {self.label_encoder_path}")
        with open(self.label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        logger.info(f"Label encoder loaded. Classes: {self.label_encoder.classes_}")
        
    def _establish_serial_connection(self) -> None:
        try:
            self.serial_conn = serial.Serial(
                self.serial_port, 
                self.baud_rate, 
                timeout=1
            )
            logger.info(f"Serial connection established: {self.serial_port}")
        except serial.SerialException as e:
            logger.error(f"Failed to establish serial connection: {e}")
            raise
            
    def parse_telemetry(self, raw_json: str) -> Optional[np.ndarray]:
        try:
            data = json.loads(raw_json)
            features = np.array([
                data['bio_signal_V'],
                data['environment']['temp_C'],
                data['environment']['humidity_pct'],
                data['environment']['pressure_hPa'],
                data['environment']['light_lux'],
                data['soil']['moisture_pct']
            ], dtype=np.float32)
            return features
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Telemetry parse error: {e}")
            return None
            
    def predict(self) -> Tuple[str, float]:
        X = np.array([list(self.feature_buffer)])
        y_pred_prob = self.model.predict(X, verbose=0)
        y_pred_class = np.argmax(y_pred_prob, axis=1)[0]
        confidence = float(np.max(y_pred_prob))
        
        predicted_label = self.label_encoder.inverse_transform([y_pred_class])[0]
        return predicted_label, confidence
        
    def run_inference_loop(self) -> None:
        logger.info("Starting inference loop...")
        sample_count = 0
        
        try:
            while True:
                if self.serial_conn.in_waiting > 0:
                    line = self.serial_conn.readline().decode('utf-8').strip()
                    
                    if not line:
                        continue
                        
                    features = self.parse_telemetry(line)
                    if features is None:
                        continue
                        
                    self.feature_buffer.append(features)
                    sample_count += 1
                    
                    logger.info(
                        f"Sample {sample_count}: "
                        f"Bio={features[0]:.4f}V "
                        f"Temp={features[1]:.1f}°C "
                        f"Soil={features[5]:.1f}%"
                    )
                    
                    if len(self.feature_buffer) == self.window_size:
                        predicted_class, confidence = self.predict()
                        
                        logger.info(
                            f"PREDICTION: {predicted_class} "
                            f"(Confidence: {confidence*100:.2f}%)"
                        )
                        
        except KeyboardInterrupt:
            logger.info("Inference loop terminated by user.")
        except Exception as e:
            logger.error(f"Runtime error: {e}")
        finally:
            self.cleanup()
            
    def cleanup(self) -> None:
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            logger.info("Serial connection closed.")


def main():
    MODELS_DIR = '../models'
    DATASETS_DIR = '../datasets'
    
    engine = PlantHealthInferenceEngine(
        model_path=os.path.join(MODELS_DIR, 'best_cnn_lstm_model.keras'),
        label_encoder_path=os.path.join(DATASETS_DIR, 'label_encoder.pkl'),
        serial_port='COM3',
        baud_rate=115200,
        window_size=10
    )
    
    engine.initialize()
    engine.run_inference_loop()


if __name__ == "__main__":
    main()
