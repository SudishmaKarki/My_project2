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

def load_data(filepath='data/RestaurantData.csv'):
    """
    Loads the dataset from a CSV file while handling NA values.
    Converts the 'Timestamp' column to datetime and sets it as the index.
    
    Parameters:
        filepath (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    df = pd.read_csv(filepath, na_values=['Na', '?'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)
    return df

def preprocess_data(df):
    """
    Preprocesses the DataFrame:
    - Prints initial inspection info.
    - Reindexes the DataFrame to ensure 24 points per day (hourly frequency).
    - Interpolates missing values in 'CustomerCount'.
    
    Parameters:
        df (pd.DataFrame): The DataFrame loaded by load_data().
        
    Returns:
        pd.DataFrame: Preprocessed DataFrame with a complete hourly index.
    """
    # Print the entire DataFrame (just for initial inspection)
    pd.set_option('display.max_rows', None)  # all rows
    print(df.head())
    print(" ")
    
    # Get DataFrame information to check data types and non-null counts
    print("\nDataFrame Info:")
    print(df.info())
    
    # Checking for missing data in all columns
    print("Missing data in each column:")
    print(df.isnull().any())
    print(" ")
    
    # Generate summary statistics for numerical columns
    print("\nSummary Statistics:")
    print(df.describe())
    
    print(df['Season'].value_counts())
    
    # Look for 24 Points per Day
    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='h')
    df_full = df.reindex(full_index)
    df_full.index.name = 'Timestamp'
    
    # Interpolate missing values 
    df_full['CustomerCount'] = df_full['CustomerCount'].interpolate(method='time')
    print(df_full.groupby(df_full.index.date).size().head())
    
    return df_full

# For testing purposes
if __name__ == '__main__':
    df = load_data()
    df_full = preprocess_data(df)

  # Print the final DataFrame preview if data was successfully loaded
    if df_full is not None:
        print("\nFinal DataFrame Preview:")
        print(df_full.head())