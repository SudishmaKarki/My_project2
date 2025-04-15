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

                         ##BASELINE MODEL SARIMAX##

def prepare_sarimax_data(restaurant_train, restaurant_test):
    restaurant_subset_train = restaurant_train.copy()
    restaurant_subset_test = restaurant_test.copy()

    restaurant_subset_train.index = pd.to_datetime(restaurant_subset_train.index)
    restaurant_subset_train = restaurant_subset_train.asfreq('h')

    restaurant_subset_test.index = pd.to_datetime(restaurant_subset_test.index)
    restaurant_subset_test = restaurant_subset_test.asfreq('h')

    train_series = restaurant_subset_train['CustomerCount']
    test_series = restaurant_subset_test['CustomerCount']

    return train_series, test_series

def check_stationarity(timeseries):
    result = adfuller(timeseries, autolag='AIC')
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    print("Result: Stationary" if result[1] < 0.05 else "Result: Non-Stationary")

def plot_acf_pacf(timeseries, lags=40, title_prefix=""):
    plot_acf(timeseries.dropna(), lags=lags)
    plt.title(f"{title_prefix} ACF")
    plt.show()

    plot_pacf(timeseries.dropna(), lags=lags)
    plt.title(f"{title_prefix} PACF")
    plt.show()

def fit_sarimax_model(train_series):
    p, d, q = 1, 1, 1
    P, D, Q, s = 1, 1, 1, 24  # Hourly data with daily seasonality
    model_sarimax = SARIMAX(train_series, order=(p, d, q), seasonal_order=(P, D, Q, s))
    results_sarimax = model_sarimax.fit(disp=False)
    print(results_sarimax.summary())
    return results_sarimax

def analyze_residual_spike(residuals, original_df):
    largest_residual_timestamp = residuals.abs().idxmax()
    largest_residual_value = residuals.loc[largest_residual_timestamp]
    print("Largest residual at:", largest_residual_timestamp)
    print("Residual value:", largest_residual_value)
    print("\nData at the timestamp with the largest residual:")
    print(original_df.loc[largest_residual_timestamp])
    return largest_residual_timestamp, largest_residual_value

def ljung_box_test(residuals, lags=[10]):
    print("Ljung-Box test results:")
    lb_test_result = acorr_ljungbox(residuals, lags=lags, return_df=True)
    print(lb_test_result)
    return lb_test_result

#Test set forcasting
def forecast_sarimax_model(results_sarimax, n_test, test_index=None):
    forecast_obj = results_sarimax.get_forecast(steps=n_test)
    forecast_mean = forecast_obj.predicted_mean
    forecast_ci = forecast_obj.conf_int()

    if test_index is not None:
        forecast_mean.index = test_index
        forecast_ci.index = test_index

    return forecast_mean, forecast_ci

def identify_peak_hours_sarimax(forecast_mean, test_series, threshold_ratio=0.6):
    forecast_df = forecast_mean.to_frame(name='yhat')
    forecast_df['Hour'] = forecast_df.index.hour

    test_df = test_series.to_frame(name='y')
    test_df['Hour'] = test_df.index.hour

    hourly_avg_forecast = forecast_df.groupby('Hour')['yhat'].mean()
    threshold = threshold_ratio * hourly_avg_forecast.max()
    peak_hours_dynamic = sorted([hour for hour, avg in hourly_avg_forecast.items() if avg >= threshold])

    forecast_peak = forecast_df[forecast_df['Hour'].isin(peak_hours_dynamic)]
    test_peak = test_df[test_df['Hour'].isin(peak_hours_dynamic)]

    return peak_hours_dynamic, threshold, hourly_avg_forecast, forecast_peak, test_peak


def evaluate_sarimax_metrics(test_series, forecast_mean, test_peak, forecast_peak):
    metrics = {}

    # Overall Metrics
    mae_all = mean_absolute_error(test_series, forecast_mean)
    rmse_all = np.sqrt(mean_squared_error(test_series, forecast_mean))
    mape_all = mean_absolute_percentage_error(test_series, forecast_mean)

    metrics["overall"] = {
        "MAE": mae_all,
        "RMSE": rmse_all,
        "MAPE": mape_all
    }

    # Peak Hour Metrics
    mae_peak = mean_absolute_error(test_peak['y'], forecast_peak['yhat'])
    rmse_peak = np.sqrt(mean_squared_error(test_peak['y'], forecast_peak['yhat']))
    mape_peak = mean_absolute_percentage_error(test_peak['y'], forecast_peak['yhat'])

    metrics["peak_hours metrics"] = {
        "MAE": mae_peak,
        "RMSE": rmse_peak,
        "MAPE": mape_peak
    }

    return metrics

def rolling_forecast_sarimax(train_series, test_series, order, seasonal_order, peak_hours_dynamic, window_size=500, step=5, forecast_steps=1, max_points=50):
  
    test_series_small = test_series.iloc[:max_points]
    data = pd.concat([train_series, test_series_small])

    rolling_forecasts_overall = []
    rolling_actuals_overall = []
    rolling_forecasts_peak = []
    rolling_actuals_peak = []

    for i in range(0, len(test_series_small), step):
        train_window = data.iloc[i : i + window_size]
        model = SARIMAX(train_window, order=order, seasonal_order=seasonal_order)
        results = model.fit(disp=False)

        forecast = results.forecast(steps=forecast_steps)
        forecast_value = forecast.iloc[0]
        forecast_time = test_series_small.index[i]

        rolling_forecasts_overall.append(forecast_value)
        rolling_actuals_overall.append(test_series_small.iloc[i])

        if forecast_time.hour in peak_hours_dynamic:
            rolling_forecasts_peak.append(forecast_value)
            rolling_actuals_peak.append(test_series_small.iloc[i])

    processed_indices = test_series_small.index[::step]
    rolling_forecasts_overall = pd.Series(rolling_forecasts_overall, index=processed_indices)
    rolling_actuals_overall = pd.Series(rolling_actuals_overall, index=processed_indices)

    # Align peak index based on hour match
    peak_index = [ts for ts in processed_indices if ts.hour in peak_hours_dynamic]
    rolling_forecasts_peak = pd.Series(rolling_forecasts_peak, index=peak_index)
    rolling_actuals_peak = pd.Series(rolling_actuals_peak, index=peak_index)

    overall_metrics = {
        'MAE': mean_absolute_error(rolling_actuals_overall, rolling_forecasts_overall),
        'RMSE': np.sqrt(mean_squared_error(rolling_actuals_overall, rolling_forecasts_overall)),
        'MAPE': mean_absolute_percentage_error(rolling_actuals_overall, rolling_forecasts_overall)
    }

    peak_metrics = {'MAE': None, 'RMSE': None, 'MAPE': None}
    if len(rolling_forecasts_peak) > 0:
        peak_metrics = {
            'MAE': mean_absolute_error(rolling_actuals_peak, rolling_forecasts_peak),
            'RMSE': np.sqrt(mean_squared_error(rolling_actuals_peak, rolling_forecasts_peak)),
            'MAPE': mean_absolute_percentage_error(rolling_actuals_peak, rolling_forecasts_peak)
        }

    return overall_metrics, peak_metrics, rolling_forecasts_overall, rolling_actuals_overall, rolling_forecasts_peak, rolling_actuals_peak

def generate_future_forecast_sarimax(results, periods=30*24):
    forecast = results.get_forecast(steps=periods)
    forecast_df = forecast.predicted_mean.to_frame(name='yhat')
    forecast_df['ds'] = forecast_df.index
    forecast_df['Hour'] = forecast_df['ds'].dt.hour
    return forecast_df

def group_forecast_by_hour(forecast_df, threshold_ratio=0.6):
    hourly_avg = forecast_df.groupby('Hour')['yhat'].mean().round(2)
    threshold = threshold_ratio * hourly_avg.max()
    peak_hours = sorted([hour for hour, val in hourly_avg.items() if val >= threshold])
    hourly_df = hourly_avg.reset_index(name='Avg Forecast (yhat)')
    return hourly_df, threshold, peak_hours

                                  #SARIMAX Grid search for model refinement 1
def sarimax_grid_search(train_series):
    pdq = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]               
    seasonal_pdq = [(1, 0, 0, 24), (1, 1, 0, 24), (1, 1, 1, 24)]  

    results_list = []

    print("Starting SARIMAX Grid Search...")
    for order in pdq:
        for seasonal_order in seasonal_pdq:
            try:
                model = SARIMAX(train_series,
                                order=order,
                                seasonal_order=seasonal_order,
                                enforce_stationarity=False,
                                enforce_invertibility=False)
                results = model.fit(disp=False, maxiter=200)
                results_list.append({
                    'order': order,
                    'seasonal_order': seasonal_order,
                    'AIC': results.aic,
                    'BIC': results.bic
                })
                print(f"SARIMAX{order}x{seasonal_order} - AIC: {results.aic:.2f}, BIC: {results.bic:.2f}")
                del model, results 
                gc.collect()
            except Exception as e:
                print(f"SARIMAX{order}x{seasonal_order} failed: {e}")
                continue

    results_df = pd.DataFrame(results_list)
    best_params = results_df.loc[results_df['AIC'].idxmin()]

    print("\nBest SARIMAX Parameters Based on AIC:")
    print(best_params)

    return results_df, best_params

def retrain_sarimax_model(train_series, best_order, best_seasonal_order):
    model_sarimax_best = SARIMAX(
        train_series,
        order=best_order,
        seasonal_order=best_seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    results_sarimax_best = model_sarimax_best.fit(disp=False)
    print(results_sarimax_best.summary())
    return results_sarimax_best


def ljung_box_test_residuals(residuals_refined, lags=[10]):
    lb_test_results = acorr_ljungbox(residuals_refined.dropna(), lags=lags, return_df=True)
    print("Ljung-Box Test Results (Refined SARIMAX Residuals):")
    print(lb_test_results)
    return lb_test_results

#Test set forcasting
def forecast_with_refined_sarimax(results_sarimax_best, test_series):
    n_test = len(test_series)
    forecast_obj = results_sarimax_best.get_forecast(steps=n_test)
    forecast_mean = forecast_obj.predicted_mean
    forecast_ci = forecast_obj.conf_int()

    forecast_mean.index = test_series.index
    forecast_ci.index = test_series.index

    return forecast_mean, forecast_ci

def analyze_peak_hours_sarimax_refined(forecast_mean, test_series, threshold_ratio=0.6):
    forecast_df = forecast_mean.to_frame(name='yhat')
    forecast_df['Hour'] = forecast_df.index.hour

    test_df = test_series.to_frame(name='y')
    test_df['Hour'] = test_df.index.hour

    hourly_avg_forecast = forecast_df.groupby('Hour')['yhat'].mean()
    threshold = threshold_ratio * hourly_avg_forecast.max()
    peak_hours_sr1 = sorted([hour for hour, avg in hourly_avg_forecast.items() if avg >= threshold])

    forecast_peak = forecast_df[forecast_df['Hour'].isin(peak_hours_sr1)]
    test_peak = test_df[test_df['Hour'].isin(peak_hours_sr1)]

    return peak_hours_sr1, threshold, hourly_avg_forecast, forecast_peak, test_peak

def evaluate_refined_sarimax_metrics(test_series, forecast_mean, test_peak, forecast_peak):
    # --- Overall Metrics ---
    mae_all = mean_absolute_error(test_series, forecast_mean)
    rmse_all = np.sqrt(mean_squared_error(test_series, forecast_mean))
    mape_all = mean_absolute_percentage_error(test_series, forecast_mean)

    # --- Peak Hour Metrics ---
    mae_peak = mean_absolute_error(test_peak['y'], forecast_peak['yhat'])
    rmse_peak = np.sqrt(mean_squared_error(test_peak['y'], forecast_peak['yhat']))
    mape_peak = mean_absolute_percentage_error(test_peak['y'], forecast_peak['yhat'])

    return [
        ["MAE", "Overall", mae_all],
        ["RMSE", "Overall", rmse_all],
        ["MAPE", "Overall", mape_all],
        ["MAE", "Peak Hours", mae_peak],
        ["RMSE", "Peak Hours", rmse_peak],
        ["MAPE", "Peak Hours", mape_peak],
    ]

def rolling_forecast_sarimax_refined(train_series, test_series, best_order, best_seasonal_order, peak_hours, window_size=500, step=5, forecast_steps=1, max_points=50):
    test_series_small = test_series.iloc[:max_points]
    data = pd.concat([train_series, test_series_small])

    rolling_forecasts_overall = []
    rolling_actuals_overall = []
    rolling_forecasts_peak = []
    rolling_actuals_peak = []

    for i in range(0, len(test_series_small), step):
        train_window = data.iloc[i : i + window_size]
        model = SARIMAX(train_window, order=best_order, seasonal_order=best_seasonal_order)
        results = model.fit(disp=False)

        forecast = results.forecast(steps=forecast_steps)
        forecast_value = forecast.iloc[0]
        forecast_time = test_series_small.index[i]

        rolling_forecasts_overall.append(forecast_value)
        rolling_actuals_overall.append(test_series_small.iloc[i])

        if forecast_time.hour in peak_hours:
            rolling_forecasts_peak.append(forecast_value)
            rolling_actuals_peak.append(test_series_small.iloc[i])

    processed_indices = test_series_small.index[::step]
    rolling_forecasts_overall = pd.Series(rolling_forecasts_overall, index=processed_indices)
    rolling_actuals_overall = pd.Series(rolling_actuals_overall, index=processed_indices)

    peak_index = [ts for ts in processed_indices if ts.hour in peak_hours]
    rolling_forecasts_peak = pd.Series(rolling_forecasts_peak, index=peak_index)
    rolling_actuals_peak = pd.Series(rolling_actuals_peak, index=peak_index)

    overall_metrics = {
        'MAE': mean_absolute_error(rolling_actuals_overall, rolling_forecasts_overall),
        'RMSE': np.sqrt(mean_squared_error(rolling_actuals_overall, rolling_forecasts_overall)),
        'MAPE': mean_absolute_percentage_error(rolling_actuals_overall, rolling_forecasts_overall)
    }

    peak_metrics = {'MAE': None, 'RMSE': None, 'MAPE': None}
    if len(rolling_forecasts_peak) > 0:
        peak_metrics = {
            'MAE': mean_absolute_error(rolling_actuals_peak, rolling_forecasts_peak),
            'RMSE': np.sqrt(mean_squared_error(rolling_actuals_peak, rolling_forecasts_peak)),
            'MAPE': mean_absolute_percentage_error(rolling_actuals_peak, rolling_forecasts_peak)
        }

    return overall_metrics, peak_metrics, rolling_forecasts_overall, rolling_actuals_overall, rolling_forecasts_peak, rolling_actuals_peak

def forecast_future_sarimax_model_refined(results_sarimax_best, periods=30*24):
    forecast = results_sarimax_best.get_forecast(steps=periods)
    forecast_df = forecast.predicted_mean.to_frame(name='yhat')
    forecast_df['ds'] = forecast_df.index
    forecast_df['Hour'] = forecast_df['ds'].dt.hour
    return forecast_df

def future_forecast_by_hour_sarimax_refined(forecast_df, threshold_ratio=0.6):
    hourly_avg = forecast_df.groupby('Hour')['yhat'].mean().round(2)
    threshold = threshold_ratio * hourly_avg.max()
    peak_hours = sorted([hour for hour, val in hourly_avg.items() if val >= threshold])
    hourly_df = hourly_avg.reset_index(name='Avg Forecast (yhat)')
    return hourly_df, threshold, peak_hours

                                   #SARIMAX Exogenous Variables model refinement 

#Exogenous Variable Preparation
def create_exogenous_variables(train_df, test_df):
    holiday_dummies_train = pd.get_dummies(train_df['Holiday'], prefix='Holiday', drop_first=False)
    holiday_dummies_test = pd.get_dummies(test_df['Holiday'], prefix='Holiday', drop_first=False)

    all_cols = holiday_dummies_train.columns.union(holiday_dummies_test.columns)
    holiday_dummies_train = holiday_dummies_train.reindex(columns=all_cols, fill_value=0).astype(float)
    holiday_dummies_test = holiday_dummies_test.reindex(columns=all_cols, fill_value=0).astype(float)

    # ✅ FIX: Use pd.Series with index to get hour values as a proper Series
    hour_train = pd.Series(train_df.index.hour, index=train_df.index, name='hour')
    hour_test = pd.Series(test_df.index.hour, index=test_df.index, name='hour')

    exog_train = pd.concat([hour_train, holiday_dummies_train], axis=1)
    exog_test = pd.concat([hour_test, holiday_dummies_test], axis=1)

    return exog_train, exog_test

#Fit SARIMAX with Exogenous Variables
def fit_sarimax_with_exog(train_series, exog_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 24), show_summary=True):
    model_exog_full = SARIMAX(train_series, exog=exog_train, order=order, seasonal_order=seasonal_order)
    results_exog_full = model_exog_full.fit(disp=False, maxiter=500, pgtol=1e-8, method='lbfgs', cov_type='robust')
    if show_summary:
        print(results_exog_full.summary())
    return results_exog_full


#Ljung-Box Test
def ljung_box_test_refined_sarimax(results_exog_full):
    lb_test = acorr_ljungbox(results_exog_full.dropna(), lags=[10], return_df=True)
    print("Ljung-Box test results for refined model residuals:")
    print(lb_test)
    return lb_test

#Investigate Largest Residual (Refined SARIMAX)
def analyze_largest_residual_sarimax_exog(residuals_exog, original_df):
    largest_residual_timestamp_exog = residuals_exog.abs().idxmax()
    largest_residual_value_exog = residuals_exog.loc[largest_residual_timestamp_exog]
    
    print("Largest residual at:", largest_residual_timestamp_exog, 
          "with value:", largest_residual_value_exog)
    print("Data at that timestamp:")
    print(original_df.loc[largest_residual_timestamp_exog])
    
    return largest_residual_timestamp_exog, largest_residual_value_exog

#Test set forcasting
def forecast_with_exog(results, exog_test, test_index):
    forecast_obj_exog_full = results.get_forecast(steps=len(exog_test), exog=exog_test)
    forecast_mean_exog_full = forecast_obj_exog_full.predicted_mean
    forecast_ci = forecast_obj_exog_full.conf_int()
    forecast_mean_exog_full.index = test_index
    forecast_ci.index = test_index
    return forecast_mean_exog_full, forecast_ci

#Identify Peak Hours from Forecast
def analyze_peak_hours_exog(forecast_mean_exog_full, test_series, threshold_ratio=0.6):
    forecast_df = forecast_mean_exog_full.to_frame(name='yhat')
    forecast_df['Hour'] = forecast_df.index.hour

    test_df = test_series.to_frame(name='y')
    test_df['Hour'] = test_df.index.hour

    hourly_avg_forecast = forecast_df.groupby('Hour')['yhat'].mean()
    threshold = threshold_ratio * hourly_avg_forecast.max()

    peak_hours = sorted([hour for hour, avg in hourly_avg_forecast.items() if avg >= threshold])

    forecast_peak = forecast_df[forecast_df['Hour'].isin(peak_hours)]
    test_peak = test_df[test_df['Hour'].isin(peak_hours)]

    return peak_hours, threshold, hourly_avg_forecast, forecast_peak, test_peak

#Evaluate Refined Model Metrics
def evaluate_sarimax_exog_metrics(test_series, forecast_mean_exog_full, test_peak, forecast_peak, epsilon=1e-6):
    # --- Overall Metrics ---
    mae_exog  = mean_absolute_error(test_series, forecast_mean_exog_full)
    rmse_exog = np.sqrt(mean_squared_error(test_series, forecast_mean_exog_full))
    mape_exog = np.mean(np.abs((test_series - forecast_mean_exog_full) / test_series)) * 100

    mape_mod = np.mean(np.abs((test_series - forecast_mean_exog_full) / (np.abs(test_series) + epsilon))) * 100
    smape = 100 * np.mean(2 * np.abs(forecast_mean_exog_full - test_series) / (np.abs(test_series) + np.abs(forecast_mean_exog_full)))

    # --- Peak Hour Metrics ---
    mae_peak = mean_absolute_error(test_peak['y'], forecast_peak['yhat'])
    rmse_peak = np.sqrt(mean_squared_error(test_peak['y'], forecast_peak['yhat']))
    mape_peak = mean_absolute_percentage_error(test_peak['y'], forecast_peak['yhat'])

    # --- Compile into DataFrame ---
    metrics_data = [
        ["MAE", "Overall", mae_exog],
        ["RMSE", "Overall", rmse_exog],
        ["MAPE", "Overall", mape_exog],
        ["Modified MAPE", "Overall", mape_mod],
        ["SMAPE", "Overall", smape],
        ["MAE", "Peak Hours", mae_peak],
        ["RMSE", "Peak Hours", rmse_peak],
        ["MAPE", "Peak Hours", mape_peak]
    ]
    metrics_df = pd.DataFrame(metrics_data, columns=["Metric", "Type", "Value"])
    metrics_df["Value"] = pd.to_numeric(metrics_df["Value"], errors='coerce')  # Ensure numeric
    metrics_df = metrics_df.replace([np.inf, -np.inf], np.nan)  # Replace infs
    metrics_df = metrics_df.fillna(0)  # ✅ Fill missing with 0 to avoid styling errors
    
    return metrics_df.style.set_caption("Refined SARIMAX (Exog): Evaluation Metrics")\
    .background_gradient(cmap='Blues', subset=["Value"])

# Rolling forcast with refined exog
def rolling_forecast_sarimax_exog(
    train_series,
    test_series,
    exog_train,
    exog_test,
    best_order,
    best_seasonal_order,
    peak_hours,
    window_size=500,
    step=5,
    forecast_steps=1,
    max_points=50
):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
    import numpy as np
    import pandas as pd
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    # Trim test set for faster rolling
    test_series_small = test_series.iloc[:max_points]
    exog_test_small = exog_test.loc[test_series_small.index]
    
    # Combine data for rolling window
    data = pd.concat([train_series, test_series_small])
    exog_full = pd.concat([exog_train, exog_test_small])

    # Initialize results
    rolling_forecasts_overall = []
    rolling_actuals_overall = []
    rolling_forecasts_peak = []
    rolling_actuals_peak = []

    convergence_issues = 0
    failed_iterations = []

    # Rolling forecast loop
    for i in range(0, len(test_series_small), step):
        train_window = data.iloc[i : i + window_size]
        exog_window = exog_full.iloc[i : i + window_size]
        exog_forecast = exog_test_small.iloc[i : i + forecast_steps]

        model = SARIMAX(
            train_window,
            exog=exog_window,
            order=best_order,
            seasonal_order=best_seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        results = model.fit(disp=False, method='lbfgs', maxiter=2000)

        if not results.mle_retvals.get('converged', True):
            print(f"⚠️ Model failed to converge at iteration {i}")
            convergence_issues += 1
            failed_iterations.append(i)

        forecast = results.forecast(steps=forecast_steps, exog=exog_forecast)
        forecast_value = forecast.iloc[0]
        forecast_time = test_series_small.index[i]

        rolling_forecasts_overall.append(forecast_value)
        rolling_actuals_overall.append(test_series_small.iloc[i])

        if forecast_time.hour in peak_hours:
            rolling_forecasts_peak.append(forecast_value)
            rolling_actuals_peak.append(test_series_small.iloc[i])

    # Format output
    processed_indices = test_series_small.index[::step]
    rolling_forecasts_overall = pd.Series(rolling_forecasts_overall, index=processed_indices)
    rolling_actuals_overall = pd.Series(rolling_actuals_overall, index=processed_indices)

    peak_index = [ts for ts in processed_indices if ts.hour in peak_hours]
    rolling_forecasts_peak = pd.Series(rolling_forecasts_peak, index=peak_index)
    rolling_actuals_peak = pd.Series(rolling_actuals_peak, index=peak_index)

    # Metrics
    overall_metrics = {
        'MAE': mean_absolute_error(rolling_actuals_overall, rolling_forecasts_overall),
        'RMSE': np.sqrt(mean_squared_error(rolling_actuals_overall, rolling_forecasts_overall)),
        'MAPE': mean_absolute_percentage_error(rolling_actuals_overall, rolling_forecasts_overall)
    }

    peak_metrics = {'MAE': None, 'RMSE': None, 'MAPE': None}
    if len(rolling_forecasts_peak) > 0:
        peak_metrics = {
            'MAE': mean_absolute_error(rolling_actuals_peak, rolling_forecasts_peak),
            'RMSE': np.sqrt(mean_squared_error(rolling_actuals_peak, rolling_forecasts_peak)),
            'MAPE': mean_absolute_percentage_error(rolling_actuals_peak, rolling_forecasts_peak)
        }

    # Final report
    print(f"\n✅ Rolling forecast completed.")
    print(f"⚠️ Total convergence failures: {convergence_issues}/{len(test_series_small[::step])}")
    if convergence_issues > 0:
        print(f"🔁 Failed at iterations: {failed_iterations}")

    return overall_metrics, peak_metrics, rolling_forecasts_overall, rolling_actuals_overall, rolling_forecasts_peak, rolling_actuals_peak


def create_future_exog(df, periods=30*24):
   
    # 1. Create future timestamps
    last_timestamp = df.index.max()
    future_dates = pd.date_range(start=last_timestamp + pd.Timedelta(hours=1), periods=periods, freq='h')
    future_df = pd.DataFrame(index=future_dates)
    future_df.index.name = 'ds'

    # 2. Extract hour
    future_df['hour'] = future_df.index.hour

    # 3. Generate UK holiday names
    uk_holidays = holidays.UnitedKingdom(years=range(future_df.index.min().year, future_df.index.max().year + 1))
    future_df['Holiday'] = future_df.index.date
    future_df['Holiday'] = future_df['Holiday'].apply(lambda date: uk_holidays.get(date, 'None'))

    # 4. One-hot encode holiday names to match training exog
    holiday_dummies = pd.get_dummies(future_df['Holiday'], prefix='Holiday', drop_first=False).astype(float)

    # 5. Final exog
    future_exog = pd.concat([future_df[['hour']], holiday_dummies], axis=1)

    return future_exog

#Generating a 30-day future forecast
def generate_future_forecast_sarimax_exog(results_exog_full, future_exog, periods=30*24):
    forecast = results_exog_full.get_forecast(steps=periods, exog=future_exog)
    forecast_df = forecast.predicted_mean.to_frame(name='yhat')
    forecast_df['ds'] = forecast_df.index
    return forecast_df


#Grouping forecast by hour
def group_forecast_by_hour_sarimax_exog(forecast_df, threshold_ratio=0.6):
    hourly_avg = forecast_df.groupby('Hour')['yhat'].mean().round(2)
    threshold = threshold_ratio * hourly_avg.max()
    peak_hours = sorted([hour for hour, val in hourly_avg.items() if val >= threshold])
    hourly_df = hourly_avg.reset_index(name='Avg Forecast (yhat)')
    return hourly_df, threshold, peak_hours



