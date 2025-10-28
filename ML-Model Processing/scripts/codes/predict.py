import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

models_dir = '../models'
datasets_dir = '../datasets'
realtime_csv = '../datasets/real_time_data.csv'
timesteps = 16

model_path = os.path.join(models_dir, 'final_cnn_lstm_model.keras')
model = tf.keras.models.load_model(model_path)
with open(os.path.join(datasets_dir, 'label_encoder.pkl'), 'rb') as f:
    label_encoder = pickle.load(f)
with open(os.path.join(datasets_dir, 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)

def reshape_for_lstm(X, timesteps):
    n_samples, n_features = X.shape
    features_per_step = n_features // timesteps
    if n_features % timesteps != 0:
        pad_size = timesteps * (n_features // timesteps + 1) - n_features
        X = np.pad(X, ((0, 0), (0, pad_size)), mode='constant', constant_values=0)
        features_per_step = X.shape[1] // timesteps
    X_reshaped = X.reshape(n_samples, timesteps, features_per_step)
    return X_reshaped

def generate_status_text(disease_class, confidence):
    dcstr = str(disease_class).lower()
    if 'healthy' in dcstr or 'normal' in dcstr:
        if confidence >= 85:
            return f"healthy, {confidence:.1f}%"
        elif confidence >= 70:
            return f"healthy, {confidence:.1f}%"
        else:
            return f"maybe healthy, {confidence:.1f}%"
    elif 'disease' in dcstr or 'blight' in dcstr or 'rust' in dcstr:
        if confidence >= 85:
            return f"{disease_class}, {confidence:.1f}%"
        elif confidence >= 70:
            return f"{disease_class}, {confidence:.1f}%"
        else:
            return f"{disease_class}? {confidence:.1f}%"
    else:
        if confidence >= 85:
            return f"{disease_class} {confidence:.1f}%"
        elif confidence >= 70:
            return f"{disease_class} {confidence:.1f}%"
        else:
            return f"{disease_class} not sure {confidence:.1f}%"

def predict_from_csv(csv_file_path, output_file=None):
    df = pd.read_csv(csv_file_path)
    if 'sample_id' in df.columns:
        sample_ids = df['sample_id'].values
        X = df.drop(columns=['sample_id']).values.astype('float32')
    else:
        sample_ids = np.arange(len(df))
        X = df.values.astype('float32')
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X_scaled = scaler.transform(X)
    X_lstm = reshape_for_lstm(X_scaled, timesteps=timesteps)
    predictions = model.predict(X_lstm, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    predicted_confidence = np.max(predictions, axis=1) * 100
    predicted_labels = label_encoder.inverse_transform(predicted_classes)
    results = []
    for sample_id, pred_class, confidence in zip(sample_ids, predicted_labels, predicted_confidence):
        status_text = generate_status_text(pred_class, confidence)
        results.append({
            'sample_id': sample_id,
            'predicted_class': pred_class,
            'confidence_%': round(confidence, 2),
            'status_text': status_text
        })
    results_df = pd.DataFrame(results)
    if output_file is None:
        output_file = csv_file_path.replace('.csv', '_predictions.csv')
    results_df.to_csv(output_file, index=False)
    print(results_df.head(5).to_string(index=False))

predict_from_csv(realtime_csv)
