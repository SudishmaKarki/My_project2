# %%
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

import sys
import os

# Add the root project directory to Python's path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.data_interpolation import load_data, preprocess_data, split_train_test

from models.model_prophet import (
    tune_prophet_model_r2,
    forecast_with_model_r2,
    select_peak_hours_r2,
    forecast_future_with_model_r2
)

# 1. Load and preprocess data 
# --- 1. Load raw CSV data ---
df_raw = load_data(filepath='../data/RestaurantData.csv')
df_full = preprocess_data(df_raw)
restaurant_train, restaurant_test = split_train_test(df_full)

#Just testing to ensure that it works
print("Train Range:", restaurant_train.index.min(), "to", restaurant_train.index.max())
print("Test Range:", restaurant_test.index.min(), "to", restaurant_test.index.max())
print("Train Shape:", restaurant_train.shape)
print("Test Shape:", restaurant_test.shape)
# %%


# %%
