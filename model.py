import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# --- 1. Load Preprocessed Data ---
# Assume your target column is 'units_sold'
df = pd.read_csv("C:\\Users\\manoj\\Downloads\\ecommerce_inventory_demand_cleaned.csv")
# --- 2. Convert to Weekly Aggregated Demand ---
df['date'] = pd.to_datetime(df['date'])
df['week'] = df['date'].dt.to_period('W')
weekly_data = df.groupby('week', sort=True)['units_sold'].sum().reset_index()
data = weekly_data['units_sold'].values.reshape(-1, 1)

# --- 3. Create Sequences ---
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

# --- 4. Train/Test Split ---
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# --- 5. Build LSTM Model ---
model = Sequential([
    LSTM(32, return_sequences=True, input_shape=(time_steps, 1)),
    LSTM(16),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.003), loss=tf.keras.losses.Huber())
model.summary()

# --- 6. Train Model ---
history = model.fit(X_train, y_train, epochs=250, batch_size=8,
                    validation_data=(X_test, y_test), verbose=1,
                    shuffle=False, callbacks=[tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss', patience=35, restore_best_weights=True)])

# --- 7. Save Model for Testing Script ---
model.save("lstm_model.h5")
print("✅ Model trained and saved as lstm_model.h5")
