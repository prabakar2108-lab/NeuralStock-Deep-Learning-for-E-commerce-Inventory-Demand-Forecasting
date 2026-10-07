import pandas as pd
import openpyxl

# Load cleaned dataset
df = pd.read_csv("C:/Users/manoj/Downloads/ecommerce_inventory_demand_cleaned.csv")

# --- 1. Revenue Feature ---
df['revenue'] = df['units_sold'] * df['unit_price']

# --- 2. Discount Flag ---
df['has_discount'] = df['discount_pct'].apply(lambda x: 1 if x > 0 else 0)

# --- 3. Promotion-Discount Interaction ---
df['promo_discount_effect'] = df['is_promotion'] * df['discount_pct']

# --- 4. Lag Features (previous demand) ---
df['lag_1'] = df['units_sold'].shift(1)   # yesterday’s sales
df['lag_7'] = df['units_sold'].shift(7)   # last week’s sales

# --- 5. Rolling Statistics ---
df['rolling_mean_7'] = df['units_sold'].rolling(window=7).mean()
df['rolling_std_7'] = df['units_sold'].rolling(window=7).std()

# --- 6. Categorical Encoding ---
df = pd.get_dummies(df, columns=['product_category', 'day_of_week', 'month'], drop_first=True)

# --- 7. Stock Utilization ---
df['stock_utilization'] = df['units_sold'] / (df['stock_on_hand'] + 1)

# --- 8. Demand Ratio vs Reorder Point ---
df['demand_vs_reorder'] = df['units_sold'] / (df['reorder_point'] + 1)

# --- 9. Supplier Lead Time Impact ---
df['lead_time_pressure'] = df['supplier_lead_days'] * df['units_sold']
df = df.dropna(subset=['lag_1','lag_7','rolling_mean_7','rolling_std_7'])

# --- Final Check ---
print(df.head())
df.to_csv("C:/Users/manoj/Downloads/ecommerce_inventory_demand_features.csv", index=False)
df.to_excel("C:/Users/manoj/Downloads/ecommerce_inventory_demand_features.xlsx", index=False)
print("\nFeature engineered data saved to CSV and Excel files.")
