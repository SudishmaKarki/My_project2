import streamlit as st
import pandas as pd
from datetime import datetime

from models.model_prophet import forecast_future_with_model_r2
from optimisation.staff_opt import reoptimize_schedule, final_schedule, future_hourly_avg_r2

st.set_page_config(page_title="Staff Scheduler", layout="wide")

st.title("🍽️ AI-Powered Restaurant Staff Scheduler")
st.markdown("This app forecasts customer demand and optimizes staffing using Genetic Algorithms.")

# Select a start date for re-optimization
start_date = st.date_input("Select a forecast start date:", value=datetime(2022, 1, 1))

# Button to run re-optimization
if st.button("Run Optimization"):
    result = reoptimize_schedule(start_date=start_date)

    st.subheader(f"📈 Forecasted Demand (Start: {start_date})")
    st.bar_chart(result['hourly_avg'])

    st.subheader("⏰ Peak Hours")
    st.write(result['peak_hours'])

    st.subheader("👥 Optimized Schedule for Peak Hours")
    peak_schedule_df = pd.DataFrame({
        'Hour': result['peak_hours'],
        'Staff Count': result['optimized_schedule']
    })
    st.dataframe(peak_schedule_df)

    st.subheader("💸 GA Optimization Cost")
    st.metric("Total Cost", result['cost'])

    st.subheader("📊 Final 24-Hour Schedule")
    full_schedule_df = pd.DataFrame({
        'Hour': list(final_schedule.keys()),
        'Staff Count': list(final_schedule.values())
    }).sort_values('Hour')
    st.dataframe(full_schedule_df)

    st.subheader("📉 Forecast vs. Staffing (Visualization)")
    st.line_chart(full_schedule_df.set_index('Hour'))

st.markdown("---")
st.markdown("Made with 💡 by Sudishma · Powered by Prophet + Genetic Algorithms")