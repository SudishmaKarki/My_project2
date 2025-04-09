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

                         ##BASELINE MODEL##
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

                     ##REFINE MODEL - HYPERPARAMETER TUNING##
# Model refinement using hyperparameter tuning with composite score from RMSE and MAE

def tune_prophet_model(train_df, test_df, param_grid):
    all_params = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]
    
    rmse_list = []
    mae_list = []
    composite_scores = []

    print("Hyperparameter Tuning (Composite = RMSE + MAE):")
    for params in all_params:
        m_tuned_r1 = Prophet(
            changepoint_prior_scale=params['changepoint_prior_scale'],
            seasonality_prior_scale=params['seasonality_prior_scale'],
            seasonality_mode=params['seasonality_mode']
        )
        m_tuned_r1.fit(train_df)
        forecast = m_tuned_r1.predict(test_df)

        rmse = np.sqrt(mean_squared_error(test_df['y'], forecast['yhat']))
        mae = mean_absolute_error(test_df['y'], forecast['yhat'])
        composite_score = rmse + mae

        rmse_list.append(rmse)
        mae_list.append(mae)
        composite_scores.append(composite_score)

        print(f"Params: {params} --> RMSE: {rmse:.4f}, MAE: {mae:.4f}, Composite: {composite_score:.4f}")

    tuning_results = pd.DataFrame(all_params)
    tuning_results['rmse'] = rmse_list
    tuning_results['mae'] = mae_list
    tuning_results['composite'] = composite_scores

    best_params = tuning_results.loc[tuning_results['composite'].idxmin()]
    print("\nBest Hyperparameters based on Composite Score (RMSE + MAE):")
    print(best_params)

    m_best_r1 = Prophet(
        changepoint_prior_scale=best_params['changepoint_prior_scale'],
        seasonality_prior_scale=best_params['seasonality_prior_scale'],
        seasonality_mode=best_params['seasonality_mode']
    )
    m_best_r1.fit(train_df)

    return m_best_r1, best_params, tuning_results

def forecast_with_model_r1(m_best_r1, restaurant_test_prophet):
    restaurant_test_fcst_best_r1 = m_best_r1.predict(restaurant_test_prophet)
    restaurant_test_fcst_best_r1['Hour'] = restaurant_test_fcst_best_r1['ds'].dt.hour
    return restaurant_test_fcst_best_r1


def select_peak_hours(
    restaurant_test_fcst_best_r1, 
    restaurant_test_prophet, 
    threshold_ratio=0.6
):
 
    # Copy forecast and extract Hour column
    df = restaurant_test_fcst_best_r1.copy()
    df['Hour'] = df['ds'].dt.hour

    # Compute hourly average forecast
    hourly_avg_best_r1 = df.groupby('Hour')['yhat'].mean()

    # Calculate threshold and identify peak hours
    threshold_best_r1 = threshold_ratio * hourly_avg_best_r1.max()
    peak_hours_dynamic_best_r1 = sorted([hour for hour, demand in hourly_avg_best_r1.items() if demand >= threshold_best_r1])

    # Filter forecast and actual test data for peak hours
    tuned_peak_fcst_dynamic_best_r1 = df[df['Hour'].isin(peak_hours_dynamic_best_r1)]
    restaurant_test_prophet_peak_dynamic_best_r1 = restaurant_test_prophet[
        restaurant_test_prophet['ds'].dt.hour.isin(peak_hours_dynamic_best_r1)
    ]

    return (peak_hours_dynamic_best_r1, threshold_best_r1,
            tuned_peak_fcst_dynamic_best_r1, restaurant_test_prophet_peak_dynamic_best_r1,
            hourly_avg_best_r1)

def evaluate_tuned_model_metrics(
    restaurant_test,
    restaurant_test_fcst_best_r1,
    restaurant_test_prophet_peak_dynamic_best_r1,
    tuned_peak_fcst_dynamic_best_r1
):
    # ----- Overall Metrics -----
    mae_all_best_r1 = mean_absolute_error(
        y_true=restaurant_test['CustomerCount'],
        y_pred=restaurant_test_fcst_best_r1['yhat']
    )
    rmse_all_best_r1 = np.sqrt(mean_squared_error(
        y_true=restaurant_test['CustomerCount'],
        y_pred=restaurant_test_fcst_best_r1['yhat']
    ))
    mape_all_best_r1 = mean_absolute_percentage_error(
        y_true=restaurant_test['CustomerCount'],
        y_pred=restaurant_test_fcst_best_r1['yhat']
    )

    print("\nTuned Model Overall Test Data Metrics:")
    print("MAE:", mae_all_best_r1)
    print("RMSE:", rmse_all_best_r1)
    print("MAPE:", mape_all_best_r1)

    # ----- Peak Hours Metrics -----
    actual_peak_best_r1 = restaurant_test_prophet_peak_dynamic_best_r1.set_index('ds')['y']
    predicted_peak_best_r1 = tuned_peak_fcst_dynamic_best_r1.set_index('ds')['yhat']

    mae_peak_best_r1 = mean_absolute_error(actual_peak_best_r1, predicted_peak_best_r1)
    rmse_peak_best_r1 = np.sqrt(mean_squared_error(actual_peak_best_r1, predicted_peak_best_r1))
    mape_peak_best_r1 = mean_absolute_percentage_error(actual_peak_best_r1, predicted_peak_best_r1)

    print("\nTuned Model Peak Hours Metrics:")
    print("MAE:", mae_peak_best_r1)
    print("RMSE:", rmse_peak_best_r1)
    print("MAPE:", mape_peak_best_r1)

    return {
        "mae_all_best_r1": mae_all_best_r1,
        "rmse_all_best_r1": rmse_all_best_r1,
        "mape_all_best_r1": mape_all_best_r1,
        "mae_peak_best_r1": mae_peak_best_r1,
        "rmse_peak_best_r1": rmse_peak_best_r1,
        "mape_peak_best_r1": mape_peak_best_r1
    }

def cross_validate_tuned_r1(m_best_r1, initial='730 days', period='180 days', horizon='365 days'):
    df_cv_r1 = cross_validation(m_best_r1, initial=initial, period=period, horizon=horizon, disable_tqdm=True)
    df_p_r1 = performance_metrics(df_cv_r1)
    return df_cv_r1, df_p_r1

         ##REFINE MODEL 2 - HYPERPARAMETER TUNING WITH HOLIDAY AND HOUR REGRESSOR ##

# Generate holidays dataframe for Prophet
def prepare_holiday_df(start_year, end_year):
    uk_holidays = holidays.UnitedKingdom(years=range(start_year, end_year + 1))
    return pd.DataFrame({
        'ds': list(uk_holidays.keys()),
        'holiday': list(uk_holidays.values())
    }).sort_values('ds').reset_index(drop=True)

# Train with holidays and hour regressor + tune

def tune_prophet_model_r2(train_df, test_df, holiday_df, param_grid):
    all_params = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]
    rmse_list, mae_list, composite_scores = [], [], []

    print("Hyperparameter Tuning with Holidays and External Regressor (hour):")
    for params in all_params:
        m_tuned_r2 = Prophet(
            holidays=holiday_df,
            changepoint_prior_scale=params['changepoint_prior_scale'],
            seasonality_prior_scale=params['seasonality_prior_scale'],
            seasonality_mode=params['seasonality_mode']
        )
        m_tuned_r2.add_regressor('hour')
        m_tuned_r2.fit(train_df)
        forecast = m_tuned_r2.predict(test_df)

        rmse = np.sqrt(mean_squared_error(test_df['y'], forecast['yhat']))
        mae = mean_absolute_error(test_df['y'], forecast['yhat'])
        composite = rmse + mae

        rmse_list.append(rmse)
        mae_list.append(mae)
        composite_scores.append(composite)

        print(f"Params: {params} --> RMSE: {rmse:.4f}, MAE: {mae:.4f}, Composite: {composite:.4f}")

    tuning_results = pd.DataFrame(all_params)
    tuning_results['rmse'] = rmse_list
    tuning_results['mae'] = mae_list
    tuning_results['composite'] = composite_scores

    best_params = tuning_results.loc[tuning_results['composite'].idxmin()]

    m_best_r2 = Prophet(
        holidays=holiday_df,
        changepoint_prior_scale=best_params['changepoint_prior_scale'],
        seasonality_prior_scale=best_params['seasonality_prior_scale'],
        seasonality_mode=best_params['seasonality_mode']
    )
    m_best_r2.add_regressor('hour')
    m_best_r2.fit(train_df)

    return m_best_r2, best_params, tuning_results

def forecast_with_model_r2(m_best_r2, restaurant_test_prophet):
    restaurant_test_fcst_best_r2 = m_best_r2.predict(restaurant_test_prophet)
    restaurant_test_fcst_best_r2['Hour'] = restaurant_test_fcst_best_r2['ds'].dt.hour
    return restaurant_test_fcst_best_r2

def select_peak_hours_r2(
    restaurant_test_fcst_best_r2, 
    restaurant_test_prophet, 
    threshold_ratio=0.6
):
   
    df = restaurant_test_fcst_best_r2.copy()
    df['Hour'] = df['ds'].dt.hour

    hourly_avg_best_r2 = df.groupby('Hour')['yhat'].mean()
    threshold_best_r2 = threshold_ratio * hourly_avg_best_r2.max()
    peak_hours_dynamic_best_r2 = sorted([hour for hour, val in hourly_avg_best_r2.items() if val >= threshold_best_r2])

    tuned_peak_fcst_dynamic_best_r2 = df[df['Hour'].isin(peak_hours_dynamic_best_r2)]
    restaurant_test_prophet_peak_dynamic_best_r2 = restaurant_test_prophet[
        restaurant_test_prophet['ds'].dt.hour.isin(peak_hours_dynamic_best_r2)
    ]

    return (
        peak_hours_dynamic_best_r2,
        threshold_best_r2,
        tuned_peak_fcst_dynamic_best_r2,
        restaurant_test_prophet_peak_dynamic_best_r2,
        hourly_avg_best_r2
    )

def evaluate_metrics_r2(
    forecast_df, actual_df, customer_col='CustomerCount', threshold_ratio=0.6
):

    # Merge actuals into forecast for consistency
    forecast = forecast_df.copy()
    forecast['Hour'] = forecast['ds'].dt.hour
    actual = actual_df.copy()

    # --- Overall Metrics ---
    mae_all = mean_absolute_error(actual[customer_col], forecast['yhat'])
    rmse_all = np.sqrt(mean_squared_error(actual[customer_col], forecast['yhat']))
    mape_all = mean_absolute_percentage_error(actual[customer_col], forecast['yhat'])

    # --- Peak Hours Only ---
    hourly_avg = forecast.groupby('Hour')['yhat'].mean()
    threshold = threshold_ratio * hourly_avg.max()
    peak_hours = sorted([h for h, val in hourly_avg.items() if val >= threshold])

    forecast_peak = forecast[forecast['Hour'].isin(peak_hours)]
    actual_peak = actual[actual['ds'].dt.hour.isin(peak_hours)]

    mae_peak = mean_absolute_error(actual_peak[customer_col], forecast_peak['yhat'])
    rmse_peak = np.sqrt(mean_squared_error(actual_peak[customer_col], forecast_peak['yhat']))
    mape_peak = mean_absolute_percentage_error(actual_peak[customer_col], forecast_peak['yhat'])

    return {
        "overall": {
            "MAE": mae_all,
            "RMSE": rmse_all,
            "MAPE": mape_all
        },
        "peak_hours": {
            "MAE": mae_peak,
            "RMSE": rmse_peak,
            "MAPE": mape_peak
        }
    }

def cross_validate_model_r2( m_best_r2, initial='730 days', period='180 days', horizon='365 days'):
    df_cv_r2 = cross_validation( m_best_r2, initial=initial, period=period, horizon=horizon, disable_tqdm=True)
    df_p_r2 = performance_metrics(df_cv_r2)
    return df_cv_r2, df_p_r2

def forecast_future_with_model_r2(m_best_r2, days=30, freq='H', threshold_ratio=0.6):
    future_r2 = m_best_r2.make_future_dataframe(periods=days * 24, freq=freq)
    future_r2['hour'] = future_r2['ds'].dt.hour

    forecast_future_r2 = m_best_r2.predict(future_r2)
    forecast_future_r2['Hour'] = forecast_future_r2['ds'].dt.hour

    future_hourly_avg_r2 = forecast_future_r2.groupby('Hour')['yhat'].mean()
    threshold_r2 = threshold_ratio * future_hourly_avg_r2.max()
    future_peak_hours_r2 = sorted([hour for hour, val in future_hourly_avg_r2.items() if val >= threshold_r2])

    return forecast_future_r2, future_hourly_avg_r2, threshold_r2, future_peak_hours_r2

