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
import gc 


def compare_model_metrics(metrics_dict, model_names):
    """
    Compare forecast metrics for multiple models.
    """
    import pandas as pd

    comparison_data = []
    for model, metrics in zip(model_names, metrics_dict):
        for metric_type, values in metrics.items():
            for metric_name, value in values.items():
                comparison_data.append([model, metric_type, metric_name, round(value, 2)])
    
    df = pd.DataFrame(comparison_data, columns=["Model", "Type", "Metric", "Value"])
    return df