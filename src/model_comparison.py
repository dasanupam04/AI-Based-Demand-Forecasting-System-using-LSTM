import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA


def evaluate_lstm(model, X_test, y_test, scaler):
    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_actual = scaler.inverse_transform(y_test)

    mae = mean_absolute_error(y_actual, y_pred)
    return mae


def evaluate_random_forest(X_train, y_train, X_test, y_test):
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train.reshape(X_train.shape[0], -1), y_train)

    preds = model.predict(X_test.reshape(X_test.shape[0], -1))
    mae = mean_absolute_error(y_test, preds)
    return mae


def evaluate_arima(series):
    model = ARIMA(series, order=(5,1,0))
    fitted = model.fit()

    preds = fitted.predict(start=len(series)-30, end=len(series)-1)
    actual = series[-30:]

    mae = mean_absolute_error(actual, preds)
    return mae


def compare_models(lstm_model, X_train, y_train, X_test, y_test, scaler, original_series):
    results = {}

    results["LSTM"] = evaluate_lstm(lstm_model, X_test, y_test, scaler)
    results["ARIMA"] = evaluate_arima(original_series.flatten())

    best_model = min(results, key=results.get)

    return results, best_model
