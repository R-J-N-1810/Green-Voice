import numpy as np
import os

input_dir = '../datasets'
timesteps = 16

X_train = np.load(os.path.join(input_dir, 'X_train.npy'))
X_val = np.load(os.path.join(input_dir, 'X_val.npy'))
X_test = np.load(os.path.join(input_dir, 'X_test.npy'))

def reshape_for_lstm(X, timesteps):
    n_samples, n_features = X.shape
    features_per_step = n_features // timesteps
    if n_features % timesteps != 0:
        pad_size = (timesteps * (n_features // timesteps + 1)) - n_features
        X = np.pad(X, ((0, 0), (0, pad_size)), mode='constant', constant_values=0)
        features_per_step = X.shape[1] // timesteps
    X_reshaped = X.reshape(n_samples, timesteps, features_per_step)
    return X_reshaped

X_train_lstm = reshape_for_lstm(X_train, timesteps)
X_val_lstm = reshape_for_lstm(X_val, timesteps)
X_test_lstm = reshape_for_lstm(X_test, timesteps)

np.save(os.path.join(input_dir, 'X_train_lstm.npy'), X_train_lstm)
np.save(os.path.join(input_dir, 'X_val_lstm.npy'), X_val_lstm)
np.save(os.path.join(input_dir, 'X_test_lstm.npy'), X_test_lstm)
print("lstm reshape done.")
