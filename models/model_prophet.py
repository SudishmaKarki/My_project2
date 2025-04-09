import numpy as np
import pandas as pd
import os
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
import logging
logging.basicConfig(level=logging.WARNING)
logging.disable(logging.INFO)
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import itertools
from itertools import product
import holidays
from prophet.diagnostics import cross_validation, performance_metrics
from statsmodels.tsa.statespace.sarimax import SARIMAX
import scipy.stats as stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox


# Format the data for Prophet
def prepare_prophet_data(restaurant_train, restaurant_test):
    restaurant_train_prophet = restaurant_train.reset_index().rename(columns={'Timestamp': 'ds', 'CustomerCount': 'y'})
    restaurant_test_prophet = restaurant_test.reset_index().rename(columns={'Timestamp': 'ds', 'CustomerCount': 'y'})
    return restaurant_train_prophet, restaurant_test_prophet

#  Train the Prophet Model
def train_baseline_prophet(restaurant_train_prophet):
    m = Prophet()
    m.fit(restaurant_train_prophet)
    return m

#Test set forcasting
def forecast_with_model(m, restaurant_train_prophet):
    forecast = m.predict(restaurant_train_prophet)
    forecast['Hour'] = forecast['ds'].dt.hour
    return forecast


def calculate_peak_hours(forecast_df, threshold_ratio=0.6):
    hourly_avg = forecast_df.groupby('Hour')['yhat'].mean()
    threshold = threshold_ratio * hourly_avg.max()
    peak_hours = sorted([hour for hour, val in hourly_avg.items() if val >= threshold])
    return peak_hours, hourly_avg, threshold

def evaluate_metrics(restaurant_test, forecast_df, peak_hours):
    restaurant_test = restaurant_test.copy()
    forecast_df = forecast_df.copy()
    
    # Overall metrics
    mae_all = mean_absolute_error(restaurant_test['y'], forecast_df['yhat'])
    rmse_all = np.sqrt(mean_squared_error(restaurant_test['y'], forecast_df['yhat']))
    mape_all = mean_absolute_percentage_error(restaurant_test['y'], forecast_df['yhat'])

    # Peak hour metrics
    forecast_peak = forecast_df[forecast_df['ds'].dt.hour.isin(peak_hours)]
    test_peak = restaurant_test[restaurant_test['ds'].dt.hour.isin(peak_hours)]

    actual_peak = test_peak.set_index('ds')['y']
    predicted_peak = forecast_peak.set_index('ds')['yhat']

    mae_peak = mean_absolute_error(actual_peak, predicted_peak)
    rmse_peak = np.sqrt(mean_squared_error(actual_peak, predicted_peak))
    mape_peak = mean_absolute_percentage_error(actual_peak, predicted_peak)

    return {
        'mae_all': mae_all,
        'rmse_all': rmse_all,
        'mape_all': mape_all,
        'mae_peak': mae_peak,
        'rmse_peak': rmse_peak,
        'mape_peak': mape_peak
    }

def cross_validate_baseline(m, initial, period, horizon):
 
    df_cv = cross_validation(m, initial=initial, period=period, horizon=horizon, disable_tqdm=True)
    df_p = performance_metrics(df_cv)
    return df_cv, df_p