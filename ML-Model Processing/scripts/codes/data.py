"""
STEP 1: DATA PREPARATION WITH CLASSIFICATION LABELS
====================================================
This script prepares your merged dataset for CNN-LSTM training.

BEFORE RUNNING:
1. Update 'data_path' to point to your full_merged_data.csv
2. Update 'label_col' to your disease label column name
3. Run this script first before any other steps

OUTPUT:
- X_train.npy, X_val.npy, X_test.npy (features)
- y_train.npy, y_val.npy, y_test.npy (labels)
- label_encoder.pkl (saves class mapping)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pickle
import os

# ========== CONFIGURATION ==========
# ⚠️ UPDATE THESE PATHS ACCORDING TO YOUR SYSTEM
data_path = r'C:\Users\yeshaswi\Desktop\Green-Voice\datasets\full_merged_data.csv'
label_col = 'plant_health_status'
output_dir = '../datasets'

os.makedirs(output_dir, exist_ok=True)

print("="*70)
print("STEP 1: DATA PREPARATION FOR CNN-LSTM CLASSIFICATION")
print("="*70)

# ========== LOAD DATA ==========
print(f"\n1. Loading dataset from: {data_path}")
try:
    data = pd.read_csv(data_path)
    print(f"✓ Dataset loaded successfully!")
    print(f"  Shape: {data.shape}")
    print(f"  Columns: {list(data.columns[:5])}... (showing first 5)")
except FileNotFoundError:
    print(f"✗ ERROR: File not found at {data_path}")
    print(f"  Please update 'data_path' variable to correct location")
    exit(1)

# ========== CHECK LABEL COLUMN ==========
print(f"\n2. Checking label column: '{label_col}'")
if label_col not in data.columns:
    print(f"✗ ERROR: Column '{label_col}' not found!")
    print(f"  Available columns: {list(data.columns)}")
    print(f"  Please update 'label_col' variable")
    exit(1)

data_clean = data[data[label_col].notnull()].copy()
print(f"✓ Label column found!")
print(f"  Total samples: {len(data)}")
print(f"  Samples with valid labels: {len(data_clean)}")
print(f"  Removed {len(data) - len(data_clean)} samples with missing labels")

# ========== ANALYZE CLASS DISTRIBUTION ==========
print(f"\n3. Class Distribution:")
print("-"*70)
class_counts = data_clean[label_col].value_counts()
for cls, count in class_counts.items():
    percentage = (count / len(data_clean)) * 100
    print(f"  {str(cls):30s}: {int(count):5d} samples ({percentage:5.2f}%)")

# ========== ENCODE LABELS ==========
print(f"\n4. Encoding labels to integers...")
le = LabelEncoder()
data_clean['label_encoded'] = le.fit_transform(data_clean[label_col])
num_classes = len(le.classes_)

print(f"✓ Labels encoded successfully!")
print(f"  Number of classes: {num_classes}")

print(f"\n  Class Mapping:")
for idx, cls in enumerate(le.classes_):
    print(f"  {str(cls):30s} → {idx}")

# ========== PREPARE FEATURES AND LABELS ==========
print(f"\n5. Preparing features and labels...")

cols_to_drop = ['sample_id', label_col, 'label_encoded']
cols_to_drop = [col for col in cols_to_drop if col in data_clean.columns]

X = data_clean.drop(columns=cols_to_drop).values.astype('float32')
y = data_clean['label_encoded'].values

print(f"✓ Features extracted!")
print(f"  Feature shape: {X.shape}")
print(f"  Label shape: {y.shape}")
print(f"  Features per sample: {X.shape[1]}")

nan_count = np.isnan(X).sum()
inf_count = np.isinf(X).sum()
if nan_count > 0 or inf_count > 0:
    print(f"⚠️  WARNING: Found {nan_count} NaN and {inf_count} Inf values")
    print(f"  Replacing with zeros...")
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# ========== NORMALIZE FEATURES ==========
print(f"\n6. Normalizing features to [0, 1] range...")
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
print(f"✓ Features normalized!")
print(f"  Min value: {X.min():.6f}")
print(f"  Max value: {X.max():.6f}")
print(f"  Mean value: {X.mean():.6f}")

# ========== SPLIT DATA ==========
print(f"\n7. Splitting data into train/val/test sets...")
print(f"  Split ratio: 70% train, 15% validation, 15% test")

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
)

print(f"✓ Data split complete!")
print(f"  Train set: {X_train.shape[0]} samples ({X_train.shape[0]/len(data_clean)*100:.1f}%)")
print(f"  Val set:   {X_val.shape[0]} samples ({X_val.shape[0]/len(data_clean)*100:.1f}%)")
print(f"  Test set:  {X_test.shape[0]} samples ({X_test.shape[0]/len(data_clean)*100:.1f}%)")

print(f"\n  Class distribution in each set:")
for set_name, y_set in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
    unique, counts = np.unique(y_set, return_counts=True)
    dist = ", ".join([f"{str(le.classes_[u])}:{c}" for u, c in zip(unique, counts)])
    print(f"    {set_name:6s}: {dist}")

# ========== SAVE PROCESSED DATA ==========
print(f"\n8. Saving processed data...")

np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
np.save(os.path.join(output_dir, 'X_val.npy'), X_val)
np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
np.save(os.path.join(output_dir, 'y_val.npy'), y_val)
np.save(os.path.join(output_dir, 'y_test.npy'), y_test)

print(f"✓ Saved arrays:")
print(f"  {os.path.join(output_dir, 'X_train.npy')}")
print(f"  {os.path.join(output_dir, 'X_val.npy')}")
print(f"  {os.path.join(output_dir, 'X_test.npy')}")
print(f"  {os.path.join(output_dir, 'y_train.npy')}")
print(f"  {os.path.join(output_dir, 'y_val.npy')}")
print(f"  {os.path.join(output_dir, 'y_test.npy')}")

le_path = os.path.join(output_dir, 'label_encoder.pkl')
with open(le_path, 'wb') as f:
    pickle.dump(le, f)
print(f"  {le_path}")

scaler_path = os.path.join(output_dir, 'scaler.pkl')
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"  {scaler_path}")

print("\n" + "="*70)
print("✅ STEP 1 COMPLETE!")
print("="*70)
print(f"Next step: Run 'step2_reshape_lstm.py'")
print("="*70)
