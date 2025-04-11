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
import random


import sys
import os

# Add the root project directory to Python's path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.data_interpolation import load_data, preprocess_data, split_train_test

from models.model_prophet import (
    prepare_holiday_df,
    tune_prophet_model_r2,
    forecast_with_model_r2,
    select_peak_hours_r2,
    forecast_future_with_model_r2
)

# ---------------------------
# Set Random Seeds for Reproducibility
# ---------------------------
random.seed(42)
np.random.seed(42)


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



# Step 2: Prepare Prophet-format DataFrame
restaurant_train_prophet = restaurant_train.reset_index().rename(columns={'Timestamp': 'ds', 'CustomerCount': 'y'})
restaurant_train_prophet['hour'] = restaurant_train_prophet['ds'].dt.hour

#Just testing remove later
print(restaurant_train_prophet.head())

# Step 3: Prepare holidays
start_year = restaurant_train.index.min().year
end_year = restaurant_test.index.max().year
holiday_df = prepare_holiday_df(start_year, end_year)

#Just testing remove later
print(holiday_df.head())


# Step 4: Prophet Refinement 2 model training
param_grid = {
    'changepoint_prior_scale': [0.01],
    'seasonality_prior_scale': [0.1],
    'seasonality_mode': ['additive']
}

m_best_r2, best_params, tuning_results = tune_prophet_model_r2(
    restaurant_train_prophet,
    restaurant_train_prophet,  # (use train on both if test not needed here)
    holiday_df,
    param_grid
)

#Just testing remove later
print("Best Hyperparameters:\n", best_params)
print("\nTuning Results:\n", tuning_results)

# Step 5: Forecast future and get peak hours
forecast_future_r2, future_hourly_avg_r2, threshold_r2, future_peak_hours_r2 = forecast_future_with_model_r2(
    m_best_r2,
    days=7
)

forecasted_hourly = future_hourly_avg_r2.to_dict()

#Just testing remove later
print("\nHourly Forecast Averages:")
print(future_hourly_avg_r2)

print("\nThreshold for Peak Hours:", threshold_r2)
print("\nPeak Hours Identified:", future_peak_hours_r2)

#Staff opt
# ---------------------------
# Hybrid Strategy Parameters & Cost Configuration
# ---------------------------
capacity_per_staff = 5
min_staff = 7
max_staff = 15
cost_config = {
    'under_penalty_peak': 1500,
    'over_penalty_peak': 15,
    'under_penalty_offpeak': 1000,
    'over_penalty_offpeak': 10,
    'consecutive_penalty': 500,
    'deviation_exponent': 2,
    'deviation_scale': 50,
    'satisfaction_penalty': 50,
    'budget_penalty': 1000,
    'budget_extra': 20,
    'stress_factor': 1.0  # Set to 1.0 for normal conditions; change to simulate a spike.
}

# ---------------------------
# Define GA Functions and Cost Functions
# ---------------------------
def staffing_cost_mod(staff_levels, peak_hours, forecasted_hourly, capacity_per_staff,
                      under_penalty_peak, over_penalty_peak, under_penalty_offpeak, over_penalty_offpeak,
                      consecutive_penalty, deviation_exponent, deviation_scale):
    cost = 0
    sorted_peak_hours = sorted(peak_hours)
    for i, hour in enumerate(sorted_peak_hours):
        hour = int(hour)
        predicted = forecasted_hourly.get(hour, 0)
        required = int(np.ceil(predicted / capacity_per_staff))
        staff = staff_levels[i]
        if 18 <= hour <= 21:
            under_penalty = under_penalty_peak
            over_penalty = over_penalty_peak
        else:
            under_penalty = under_penalty_offpeak
            over_penalty = over_penalty_offpeak
        if staff < required:
            linear_cost = (required - staff) * under_penalty
        else:
            linear_cost = (staff - required) * over_penalty
        deviation = abs(staff - required)
        nonlinear_cost = (deviation ** deviation_exponent) * deviation_scale
        cost += linear_cost + nonlinear_cost
    max_consecutive_allowed = 4
    consecutive = 1
    for i in range(1, len(sorted_peak_hours)):
        if sorted_peak_hours[i] == sorted_peak_hours[i-1] + 1:
            consecutive += 1
        else:
            if consecutive > max_consecutive_allowed:
                cost += (consecutive - max_consecutive_allowed) * consecutive_penalty
            consecutive = 1
    if consecutive > max_consecutive_allowed:
        cost += (consecutive - max_consecutive_allowed) * consecutive_penalty
    return cost

def detailed_staffing_cost(staff_levels, peak_hours, forecasted_hourly, capacity_per_staff,
                             under_penalty_peak, over_penalty_peak, consecutive_penalty,
                             satisfaction_penalty, budget_penalty, budget_extra):
    demand_cost = 0
    satisfaction_cost = 0
    legal_cost = 0
    sorted_peak_hours = sorted(peak_hours)
    for i, hour in enumerate(sorted_peak_hours):
        predicted = forecasted_hourly.get(hour, 0)
        required = int(np.ceil(predicted / capacity_per_staff))
        staff = staff_levels[i]
        if staff < required:
            demand_cost += (required - staff) * under_penalty_peak
        else:
            demand_cost += (staff - required) * over_penalty_peak
        satisfaction_cost += abs(staff - required) * satisfaction_penalty
    max_consecutive_allowed = 4
    consecutive = 1
    for i in range(1, len(sorted_peak_hours)):
        if sorted_peak_hours[i] == sorted_peak_hours[i-1] + 1:
            consecutive += 1
        else:
            if consecutive > max_consecutive_allowed:
                legal_cost += (consecutive - max_consecutive_allowed) * consecutive_penalty
            consecutive = 1
    if consecutive > max_consecutive_allowed:
        legal_cost += (consecutive - max_consecutive_allowed) * consecutive_penalty
    budget_cap = min_staff * len(sorted_peak_hours) + budget_extra
    total_staff = sum(staff_levels)
    if total_staff > budget_cap:
        legal_cost += (total_staff - budget_cap) * budget_penalty
    total_cost = demand_cost + satisfaction_cost + legal_cost
    return total_cost, demand_cost, satisfaction_cost, legal_cost

def create_individual(num):
    return [random.randint(min_staff, max_staff) for _ in range(num)]

def mutate(individual, num):
    idx = random.randint(0, num - 1)
    change = random.choice([-1, 1])
    individual[idx] = max(min_staff, min(max_staff, individual[idx] + change))
    return individual

def crossover(parent1, parent2, num):
    point = random.randint(1, num - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

def select(population, fitnesses, num_select):
    selected = []
    for _ in range(num_select):
        i1, i2 = random.sample(range(len(population)), 2)
        selected.append(population[i1] if fitnesses[i1] < fitnesses[i2] else population[i2])
    return selected

# ----- GA Loop for Peak Hours with Convergence Tracking -----
best_costs = []  # To track best cost per generation

population_size = 20
num_generations = 14
num_peak = len(future_peak_hours_r2)

# Initialize the population using the number of peak hours
population = [create_individual(num_peak) for _ in range(population_size)]

for generation in range(num_generations):
    # Compute fitnesses for current population
    fitnesses = [
        staffing_cost_mod(
            ind,
            future_peak_hours_r2,
            forecasted_hourly,
            capacity_per_staff,
            cost_config['under_penalty_peak'],
            cost_config['over_penalty_peak'],
            cost_config['under_penalty_offpeak'],
            cost_config['over_penalty_offpeak'],
            cost_config['consecutive_penalty'],
            cost_config['deviation_exponent'],
            cost_config['deviation_scale']
        )
        for ind in population
    ]
    best_cost = min(fitnesses)
    best_costs.append(best_cost)  # Record best cost of this generation
    best_individual = population[fitnesses.index(best_cost)]
    print(f"Generation {generation}: Best Cost = {best_cost}, Schedule = {best_individual}")
    
    # Selection and reproduction
    selected = select(population, fitnesses, population_size)
    next_population = []
    for i in range(0, population_size, 2):
        parent1 = selected[i]
        parent2 = selected[i+1] if i+1 < population_size else selected[0]
        child1, child2 = crossover(parent1, parent2, num_peak)  # Pass num_peak to crossover
        next_population.extend([child1, child2])
    population = [mutate(ind, num_peak) if random.random() < 0.2 else ind for ind in next_population]

# Plot GA Convergence
plt.figure(figsize=(8, 4))
plt.plot(best_costs, marker='o', linestyle='-')
plt.title("GA Convergence: Best Cost Over Generations")
plt.xlabel("Generation")
plt.ylabel("Best Cost")
plt.show()

# Final evaluation after GA loop
fitnesses = [
    staffing_cost_mod(
        ind,
        future_peak_hours_r2,
        forecasted_hourly,
        capacity_per_staff,
        cost_config['under_penalty_peak'],
        cost_config['over_penalty_peak'],
        cost_config['under_penalty_offpeak'],
        cost_config['over_penalty_offpeak'],
        cost_config['consecutive_penalty'],
        cost_config['deviation_exponent'],
        cost_config['deviation_scale']
    )
    for ind in population
]
best_fitness = min(fitnesses)
best_peak_schedule = population[fitnesses.index(best_fitness)]
print("\nOptimized staffing for peak hours (using GA):")
for hour, staff in zip(sorted(future_peak_hours_r2), best_peak_schedule):
    print(f"Hour {hour}: {staff} staff members")
    

# ----- Rule-Based Assignment for Off-Peak Hours -----
min_off_peak_staff = 4
full_hours = list(range(24))
off_peak_hours = [hr for hr in full_hours if hr not in future_peak_hours_r2]
off_peak_schedule = {
    hr: max(int(np.ceil(forecasted_hourly.get(hr, 0) / capacity_per_staff)), min_off_peak_staff)
    for hr in off_peak_hours
}

print("\nRule-based staffing for off-peak hours:")
for hr in sorted(off_peak_schedule.keys()):
    print(f"Hour {hr}: {off_peak_schedule[hr]} staff members")

# ----- Combine Peak and Off-Peak Schedules into a Full 24-Hour Schedule -----
final_schedule = {}
for hr, staff in zip(sorted(future_peak_hours_r2), best_peak_schedule):
    final_schedule[hr] = staff
for hr in off_peak_hours:
    final_schedule[hr] = off_peak_schedule[hr]

for hr in sorted(final_schedule.keys()):
    note = "   (Peak Hour - GA optimized)" if hr in forecast_future_r2 else ""
    print(f"Hour {hr}: {final_schedule[hr]} staff members{note}")

# ----- Display Summary Arrays -----
required_staff = np.array([int(np.ceil(forecasted_hourly.get(hr, 0) / capacity_per_staff)) for hr in range(24)])
optimized_staff = np.array([final_schedule[hr] for hr in range(24)])
print("\nRequired staff per hour based on forecast:", required_staff)
print("Optimized staff scheduling (per hour):", optimized_staff)

# ----- Compare with Actual Staffing (StaffingLevel) -----

# Aggregate actual staffing data from your test set by hour
# (Assuming your 'restaurant_test' DataFrame has a column named "StaffingLevel")
actual_staffing_by_hour = restaurant_test.groupby(restaurant_test.index.hour)['StaffingLevel'].mean()
print("Actual Staffing by Hour:")
print(actual_staffing_by_hour)

# Create comparison DataFrame for each hour (0 to 23)
comparison_df = pd.DataFrame({
    'Hour': range(24),
    'RequiredStaff': [int(np.ceil(forecasted_hourly.get(hr, 0) / capacity_per_staff)) for hr in range(24)],
    'OptimizedStaff': [final_schedule.get(hr, 0) for hr in range(24)],
    'ActualStaff': [actual_staffing_by_hour.get(hr, np.nan) for hr in range(24)]
})

print("\nStaffing Comparison:")
print(comparison_df)

# Compute the difference between Optimized and Actual Staffing
comparison_df['Difference'] = comparison_df['OptimizedStaff'] - comparison_df['ActualStaff']
print("\nDifferences between Optimized and Actual Staffing:")
print(comparison_df[['Hour', 'OptimizedStaff', 'ActualStaff', 'Difference']])

# ----- Visualization of Comparison -----
plt.figure(figsize=(10, 5))
# Bar for Optimized Staffing (shifted left)
plt.bar(comparison_df['Hour'] - 0.2, comparison_df['OptimizedStaff'], width=0.4, label='Optimized Staffing', color='green')
# Bar for Actual Staffing (shifted right)
plt.bar(comparison_df['Hour'] + 0.2, comparison_df['ActualStaff'], width=0.4, label='Actual Staffing', color='orange')
# Line plot for Required Staffing
plt.plot(comparison_df['Hour'], comparison_df['RequiredStaff'], marker='o', linestyle='-', color='blue', label='Required Staffing')
plt.xlabel("Hour of Day")
plt.ylabel("Staff Count")
plt.title("Comparison of Staffing: Required vs. Optimized vs. Actual")
plt.xticks(range(24))
plt.legend()
plt.show()

# ---------------------------
# Visualization of Forecast and Final Staffing
# ---------------------------
plt.figure(figsize=(12, 6))
plt.bar(future_hourly_avg_r2.index, future_hourly_avg_r2, color='skyblue', label="Forecasted Demand")
for i, peak in enumerate(future_peak_hours_r2):
    plt.axvline(peak, color='red', linestyle='--', label="Optimized Peak Hour" if i == 0 else "")
staffing_values = [final_schedule.get(hr, 0) for hr in range(24)]
plt.plot(range(24), staffing_values, marker='o', color='green', label="Final Staffing")
plt.xlabel("Hour of Day")
plt.ylabel("Average Forecast / Staff Count")
plt.title("Hybrid Staff Scheduling: Forecast vs. Final Staffing")
plt.legend()
plt.xticks(range(24))
plt.show()


# Example: Evaluate GA cost for the best solution using staffing_cost_mod
ga_cost = staffing_cost_mod(
    staff_levels=best_peak_schedule,
    peak_hours=future_peak_hours_r2,
    forecasted_hourly=forecasted_hourly,
    capacity_per_staff=capacity_per_staff,
    under_penalty_peak=cost_config['under_penalty_peak'],
    over_penalty_peak=cost_config['over_penalty_peak'],
    under_penalty_offpeak=cost_config['under_penalty_offpeak'],
    over_penalty_offpeak=cost_config['over_penalty_offpeak'],
    consecutive_penalty=cost_config['consecutive_penalty'],
    deviation_exponent=cost_config['deviation_exponent'],
    deviation_scale=cost_config['deviation_scale']
)

print("\nGA Optimization Cost for Best Solution:", ga_cost)

# Evaluate the detailed cost breakdown using detailed_staffing_cost
total_cost, demand_cost, satisfaction_cost, legal_cost = detailed_staffing_cost(
    staff_levels=best_peak_schedule,
    peak_hours=future_peak_hours_r2,
    forecasted_hourly=forecasted_hourly,
    capacity_per_staff=capacity_per_staff,
    under_penalty_peak=cost_config['under_penalty_peak'],
    over_penalty_peak=cost_config['over_penalty_peak'],
    consecutive_penalty=cost_config['consecutive_penalty'],
    satisfaction_penalty=cost_config['satisfaction_penalty'],
    budget_penalty=cost_config['budget_penalty'],
    budget_extra=cost_config['budget_extra']
)

print("\nDetailed Cost Breakdown for Best Solution:")
print("Total Cost:", total_cost)
print("Demand Cost:", demand_cost)
print("Satisfaction Cost:", satisfaction_cost)
print("Legal Cost:", legal_cost)


# ---------------------------
# Rolling Forecast & Re-Optimization Function
# ---------------------------
def reoptimize_schedule(start_date, forecast_period=7*24):
    # Create a future dataframe starting from the given date
    future = m_best_r2.make_future_dataframe(periods=forecast_period, freq='h')
    future = future[future['ds'] >= pd.to_datetime(start_date)]
    future['hour'] = future['ds'].dt.hour
    forecast = m_best_r2.predict(future)
    forecast['Hour'] = forecast['ds'].dt.hour
    hourly_avg = forecast.groupby('Hour')['yhat'].mean()
    
    # Identify peak hours (threshold = 60% of max hourly forecast)
    threshold_val = 0.6 * hourly_avg.max()
    peak_hours = sorted([hr for hr, val in hourly_avg.items() if val >= threshold_val])
    
    # Create dictionary of forecasted customer counts per hour
    forecasted_hr = hourly_avg.to_dict()
    
    # Optionally apply stress factor to simulate a demand spike
    for hr in peak_hours:
        if 18 <= hr <= 21:
            forecasted_hr[hr] *= cost_config['stress_factor']
    
    # Run GA optimization on the updated forecast
    num_peak = len(peak_hours)
    #pop_size = 20
    population = [create_individual(num_peak) for _ in range(population_size )]
    best_costs = []
    for generation in range(num_generations):
        fitnesses = [
            staffing_cost_mod(
                ind,
                future_peak_hours_r2,
                forecasted_hr,
                capacity_per_staff,
                cost_config['under_penalty_peak'],
                cost_config['over_penalty_peak'],
                cost_config['under_penalty_offpeak'],
                cost_config['over_penalty_offpeak'],
                cost_config['consecutive_penalty'],
                cost_config['deviation_exponent'],
                cost_config['deviation_scale']
            )
            for ind in population
        ]
        best_cost = min(fitnesses)
        best_costs.append(best_cost)
        best_individual = population[fitnesses.index(best_cost)]
        selected = select(population, fitnesses, population_size )
        next_population = []
        for i in range(0, population_size , 2):
            parent1 = selected[i]
            parent2 = selected[i+1] if i+1 < population_size  else selected[0]
            child1, child2 = crossover(parent1, parent2, num_peak)
            next_population.extend([child1, child2])
        population = [mutate(ind, num_peak) if random.random() < 0.2 else ind for ind in next_population]
    
    fitnesses = [
        staffing_cost_mod(
            ind,
            future_peak_hours_r2,
            forecasted_hr,
            capacity_per_staff,
            cost_config['under_penalty_peak'],
            cost_config['over_penalty_peak'],
            cost_config['under_penalty_offpeak'],
            cost_config['over_penalty_offpeak'],
            cost_config['consecutive_penalty'],
            cost_config['deviation_exponent'],
            cost_config['deviation_scale']
        )
        for ind in population
    ]
    best_individual = population[fitnesses.index(min(fitnesses))]
    
    return {
        'start_date': pd.to_datetime(start_date),
        'hourly_avg': hourly_avg,
        'peak_hours': peak_hours,
        'forecasted_hourly': forecasted_hr,
        'optimized_schedule': best_individual,
        'cost': min(fitnesses)
    }

# ---------------------------
# Rolling Forecast: Loop Over a Set of Start Dates
# ---------------------------
rolling_dates = pd.date_range(start='2022-01-01', periods=3, freq='D')
rolling_results = []
for date in rolling_dates:
    result = reoptimize_schedule(date)
    rolling_results.append(result)
    print(f"\nRolling Forecast Starting {date.date()}:")
    print("Peak Hours:", result['peak_hours'])
    print("Optimized Schedule:", result['optimized_schedule'])
    print("Cost:", result['cost'])

# Visualize each rolling forecast's hourly average
for res in rolling_results:
    plt.figure(figsize=(8,4))
    res['hourly_avg'].plot(kind='bar')
    plt.title(f"Forecasted Demand by Hour Starting {res['start_date'].date()}")
    plt.xlabel("Hour of Day")
    plt.ylabel("Forecasted Customer Count")
    plt.show()


# %%
