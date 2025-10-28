import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import pickle
import os

input_dir = '../datasets'
models_dir = '../models'
results_dir = '../results'
os.makedirs(models_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

X_train = np.load(os.path.join(input_dir, 'X_train_lstm.npy'))
X_val = np.load(os.path.join(input_dir, 'X_val_lstm.npy'))
X_test = np.load(os.path.join(input_dir, 'X_test_lstm.npy'))
y_train = np.load(os.path.join(input_dir, 'y_train.npy'))
y_val = np.load(os.path.join(input_dir, 'y_val.npy'))
y_test = np.load(os.path.join(input_dir, 'y_test.npy'))
with open(os.path.join(input_dir, 'label_encoder.pkl'), 'rb') as f:
    label_encoder = pickle.load(f)
num_classes = len(label_encoder.classes_)

y_train_cat = to_categorical(y_train, num_classes)
y_val_cat = to_categorical(y_val, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

input_shape = (X_train.shape[1], X_train.shape[2])
model = models.Sequential([
    layers.Conv1D(64, 3, activation='relu', input_shape=input_shape),
    layers.BatchNormalization(),
    layers.MaxPooling1D(2),
    layers.Dropout(0.3),
    layers.Conv1D(128, 3, activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling1D(2),
    layers.Dropout(0.3),
    layers.LSTM(128, return_sequences=True),
    layers.Dropout(0.4),
    layers.LSTM(64, return_sequences=False),
    layers.Dropout(0.4),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=25,
    restore_best_weights=True,
    verbose=1,
    mode='max'
)
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.25,
    patience=8,
    min_lr=1e-5,
    verbose=1,
    mode='max'
)
checkpoint = callbacks.ModelCheckpoint(
    os.path.join(models_dir, 'best_cnn_lstm_model.keras'),
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1,
    mode='max'
)

history = model.fit(
    X_train, y_train_cat,
    epochs=200,
    batch_size=16,
    validation_data=(X_val, y_val_cat),
    callbacks=[early_stop, reduce_lr, checkpoint],
    verbose=1
)

final_model_path = os.path.join(models_dir, 'final_cnn_lstm_model.keras')
model.save(final_model_path)
test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"test acc: {test_accuracy:.4f}")

y_pred_prob = model.predict(X_test[:10], verbose=0)
y_pred = np.argmax(y_pred_prob, axis=1)
for i in range(10):
    predicted_class = label_encoder.inverse_transform([y_pred[i]])[0]
    actual_class = label_encoder.inverse_transform([y_test[i]])[0]
    confidence = np.max(y_pred_prob[i]) * 100
    match = "✓" if y_pred[i] == y_test[i] else "✗"
    print(f"{match} {i+1}: pred={str(predicted_class)} ({confidence:.2f}%) actual={str(actual_class)}")
