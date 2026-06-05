import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from utils import load_data, calculate_bill, calculate_carbon

st.set_page_config(
    page_title="Smart Meter Consumption Profiler",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Smart Meter Consumption Profiler")
st.write("AI-powered dashboard to analyze electricity consumption, detect abnormal usage, forecast demand, and suggest energy-saving actions.")

df = load_data()

st.sidebar.header("Controls")
household = st.sidebar.selectbox("Select Household", df["Household_ID"].unique())

data = df[df["Household_ID"] == household].copy()

total_units = data["Consumption_kWh"].sum()
avg_usage = data["Consumption_kWh"].mean()
peak_usage = data["Consumption_kWh"].max()
estimated_bill = calculate_bill(total_units)
carbon = calculate_carbon(total_units)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Consumption", f"{total_units:.2f} kWh")
col2.metric("Average Usage", f"{avg_usage:.2f} kWh/day")
col3.metric("Estimated Bill", f"₹{estimated_bill:.2f}")
col4.metric("Carbon Footprint", f"{carbon:.2f} kg CO₂")

st.divider()

st.subheader("📈 Daily Consumption Trend")
fig = px.line(data, x="Date", y="Consumption_kWh", markers=True)
st.plotly_chart(fig, use_container_width=True)

st.subheader("🚨 Abnormal Usage Detection")
model = IsolationForest(contamination=0.15, random_state=42)
data["Anomaly"] = model.fit_predict(data[["Consumption_kWh"]])
data["Status"] = data["Anomaly"].apply(lambda x: "Anomaly" if x == -1 else "Normal")

fig2 = px.scatter(
    data,
    x="Date",
    y="Consumption_kWh",
    color="Status",
    title="Normal vs Abnormal Usage"
)
st.plotly_chart(fig2, use_container_width=True)

anomalies = data[data["Status"] == "Anomaly"]

if len(anomalies) > 0:
    st.warning("⚠️ Abnormal electricity usage detected on these days:")
    st.dataframe(anomalies[["Date", "Consumption_kWh", "Status"]])
else:
    st.success("✅ No abnormal usage detected.")

st.divider()

st.subheader("🔮 7-Day Consumption Forecast")

data["Day_Number"] = np.arange(len(data))
X = data[["Day_Number"]]
y = data["Consumption_kWh"]

lr = LinearRegression()
lr.fit(X, y)

future_days = np.arange(len(data), len(data) + 7).reshape(-1, 1)
predictions = lr.predict(future_days)

future_dates = pd.date_range(
    start=data["Date"].max() + pd.Timedelta(days=1),
    periods=7
)

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted Consumption (kWh)": predictions
})

fig3 = px.line(
    forecast_df,
    x="Date",
    y="Predicted Consumption (kWh)",
    markers=True,
    title="Next 7 Days Forecast"
)
st.plotly_chart(fig3, use_container_width=True)
st.dataframe(forecast_df)

st.divider()

st.subheader("💡 Energy Saving Recommendations")

if avg_usage > 20:
    st.error("High average usage detected. Reduce AC, heater, and heavy appliance usage during peak hours.")
elif avg_usage > 15:
    st.warning("Moderate usage detected. Switch off unused appliances and use LED lighting.")
else:
    st.success("Good energy efficiency. Your usage pattern is under control.")

if peak_usage > avg_usage * 1.8:
    st.warning("Sudden peak load detected. Check for appliance overload or unnecessary usage.")

st.info("Smart Tip: Shift washing machine, geyser, and charging loads to non-peak hours to reduce electricity cost.")

st.divider()

st.subheader("📄 Dataset Preview")
st.dataframe(data)
