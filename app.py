import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf

# --- Load Trained Model ---
model = tf.keras.models.load_model("C:\\Users\\manoj\\final project\\lstm_model.h5")

# --- Streamlit UI ---
st.title("📊 Weekly Demand Forecasting with LSTM")
st.write("Upload weekly aggregated demand data and predict next week's demand.")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload weekly demand CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:", df.head())

    # Ensure date column exists
    df['date'] = pd.to_datetime(df['date'])
    df['week'] = df['date'].dt.to_period('W')
    weekly_data = df.groupby('week')['units_sold'].sum().reset_index()

    data = weekly_data['units_sold'].values.reshape(-1,1)

    # --- Create Sequences ---
    def create_sequences(dataset, time_steps=7):
        X, y = [], []
        for i in range(len(dataset)-time_steps):
            X.append(dataset[i:i+time_steps, 0])
            y.append(dataset[i+time_steps, 0])
        return np.array(X), np.array(y)

    time_steps = 7
    X, y = create_sequences(data, time_steps)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # --- Train/Test Split ---
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # --- Predictions on Test Set ---
    y_pred = model.predict(X_test)

    # --- Forecast Next Week ---
    last_sequence = data[-time_steps:].reshape(1, time_steps, 1)
    next_week_pred = model.predict(last_sequence)

    # --- Display Results ---
    st.subheader("✅ Model Predictions")
    st.write("**Next Week Forecasted Demand:**", float(next_week_pred[0][0]))

    # --- Plot Actual vs Predicted ---
    st.line_chart(pd.DataFrame({
        "Actual": y_test.flatten(),
        "Predicted": y_pred.flatten()
    }))
