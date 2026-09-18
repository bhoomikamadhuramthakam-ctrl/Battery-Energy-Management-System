import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
import time
import random

# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(page_title="BEMS Digital Twin", layout="wide")

# -----------------------
# SIDEBAR MENU
# -----------------------
st.sidebar.title("⚡ BEMS MENU")
menu = st.sidebar.radio("Go to", ["Dashboard", "Health Prediction", "Fault Detection", "Reports"])

# -----------------------
# LOAD DATA
# -----------------------
@st.cache_data
def load_data():
    return pd.read_excel("EV_Similar_Dataset.xlsx")

df = load_data()

# Encode
df = pd.get_dummies(df)

X = df.drop(columns=["SOH"])
y = df["SOH"]

# Train Model
model = RandomForestRegressor()
model.fit(X, y)

# -----------------------
# DIGITAL TWIN (LIVE DATA SIMULATION)
# -----------------------
def get_live_data():
    voltage = random.randint(300, 400)
    current = random.randint(5, 40)
    temp = random.randint(25, 50)
    return voltage, current, temp

# -----------------------
# DASHBOARD (REAL-TIME)
# -----------------------
if menu == "Dashboard":
    st.title("🔋 Digital Twin Battery Dashboard")

    placeholder = st.empty()

    for _ in range(100):
        voltage, current, temp = get_live_data()

        # Prepare sample for prediction
        sample = X.iloc[0:1].copy()
        sample.iloc[0, 0] = voltage
        sample.iloc[0, 1] = current

        pred = model.predict(sample)[0] * 100

        with placeholder.container():
            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Battery Level", f"{round(pred,2)}%")
            col2.metric("Voltage", f"{voltage} V")
            col3.metric("Temperature", f"{temp} °C")
            col4.metric("SOH", f"{round(pred,2)}%")

            st.subheader("📊 Battery Health")

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred,
                title={'text': "SOH"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'steps': [
                        {'range': [0, 50], 'color': "red"},
                        {'range': [50, 75], 'color': "orange"},
                        {'range': [75, 100], 'color': "green"}
                    ]
                }
            ))

            st.plotly_chart(fig, use_container_width=True,
            key=str(time.time()))

            st.subheader("📈 Live Voltage & Current")

            live_df = pd.DataFrame({
                "Voltage": [voltage],
                "Current": [current]
            })

            st.line_chart(live_df)

        time.sleep(2)

# -----------------------
# HEALTH PREDICTION (MANUAL)
# -----------------------
elif menu == "Health Prediction":
    st.title("🧠 Battery Health Prediction")

    voltage = st.slider("Voltage", 300, 400, 350)
    current = st.slider("Current", 0, 50, 10)
    temp = st.slider("Temperature", 20, 60, 30)

    if st.button("Predict"):
        sample = X.iloc[0:1].copy()
        sample.iloc[0, 0] = voltage
        sample.iloc[0, 1] = current

        pred = model.predict(sample)

        st.success(f"Predicted SOH: {round(pred[0]*100,2)}%")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred[0]*100,
            title={'text': "SOH"},
            gauge={'axis': {'range': [0, 100]}}
        ))

        st.plotly_chart(fig)

# -----------------------
# FAULT DETECTION
# -----------------------
elif menu == "Fault Detection":
    st.title("⚠ Fault Detection")

    temp = st.number_input("Temperature", 20, 100, 35)
    voltage = st.number_input("Voltage", 200, 500, 350)

    if st.button("Check Fault"):
        if temp > 45:
            st.error("🔥 Overheating Detected")
        elif voltage > 420:
            st.warning("⚡ Voltage Spike")
        else:
            st.success("✅ System Normal")

# -----------------------
# REPORTS
# -----------------------
elif menu == "Reports":
    st.title("📄 Battery Reports")

    st.dataframe(df.head(20))

    if st.button("Download Report"):
        df.to_excel("report.xlsx")
        st.success("Report Downloaded!")

    st.subheader("📊 Usage Distribution")

    fig = go.Figure(data=[go.Pie(
        labels=["Charging", "Driving", "Idle"],
        values=[45, 35, 20]
    )])

    st.plotly_chart(fig)