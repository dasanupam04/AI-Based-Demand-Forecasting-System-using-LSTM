import numpy as np
from sklearn.preprocessing import MinMaxScaler

def scale_series(series):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series)
    return scaled, scaler


def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)


def time_series_split(X, y, train_ratio=0.8):
    split = int(len(X) * train_ratio)
    return X[:split], X[split:], y[:split], y[split:]
