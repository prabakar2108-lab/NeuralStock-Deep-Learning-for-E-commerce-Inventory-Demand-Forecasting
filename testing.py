import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

# --- 1. Load Preprocessed Data ---
# Training uses weekly sums of the preprocessed target, so evaluation must use
# the same aggregation and scale.
df = pd.read_csv("C:\\Users\\manoj\\Downloads\\ecommerce_inventory_demand_cleaned.csv")
df['date'] = pd.to_datetime(df['date'])
df['week'] = df['date'].dt.to_period('W')
weekly_data = df.groupby('week', sort=True)['units_sold'].sum().reset_index()
data = weekly_data['units_sold'].values.reshape(-1, 1)

# --- 2. Recreate Sequences (same time_steps as training) ---
def create_sequences(dataset, time_steps=7):
    X, y = [], []
    for i in range(len(dataset)-time_steps):
        X.append(dataset[i:i+time_steps, 0])
        y.append(dataset[i+time_steps, 0])
    return np.array(X), np.array(y)

time_steps = 7
split_index = int(0.8 * (len(data) - time_steps))
scaler = MinMaxScaler()
scaler.fit(data[:split_index + time_steps])
scaled_data = scaler.transform(data)
X, y = create_sequences(scaled_data, time_steps)
X = X.reshape(X.shape[0], X.shape[1], 1)

# --- 3. Train/Test Split (same as training) ---
X_test = X[split_index:]
y_test = y[split_index:]

# --- 4. Load Saved Model ---
model = tf.keras.models.load_model("C:\\Users\\manoj\\final project\\lstm_model.h5")

# --- 5. Predictions ---
y_pred = model.predict(X_test)

# Convert predictions and targets back to units sold for sMAPE.
y_pred_rescaled = scaler.inverse_transform(y_pred.reshape(-1, 1))
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# --- 6. Evaluation Metrics ---
rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
mape = np.mean(np.abs((y_test_rescaled - y_pred_rescaled) / y_test_rescaled)) * 100

 #sMAPE Function (skip zero-demand rows) ---
def smape(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    mask = y_true != 0   # ignore zero-demand rows
    y_true, y_pred = y_true[mask], y_pred[mask]
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

smape_val = smape(y_test_rescaled, y_pred_rescaled)

print("✅ Model Evaluation Results")
print("RMSE:", rmse)
print("MAE:", mae)
print("MAPE (non-zero demand):", smape_val, "%")
