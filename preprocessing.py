import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load dataset
df = pd.read_csv("C:/Users/manoj/Downloads/ecommerce_inventory_demand_features.csv")

# --- 1. Identify column types ---
num_cols = df.select_dtypes(include=['int64','float64']).columns
cat_cols = df.select_dtypes(include=['object','bool']).columns

# --- 2. Encode True/False categorical columns ---
# Example: columns like 'is_promotion', 'bond_requirement', 'status'
for col in cat_cols:
    if df[col].nunique() == 2:   # binary categorical
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

# --- 3. Standardize numerical columns ---
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# --- 4. Drop identifier columns (like product_id) if not needed for modeling ---
df = df.drop(columns=['product_id'], errors='ignore')

# --- Final check ---
print(df.head())
print(df.info())
df.to_csv("C:/Users/manoj/Downloads/ecommerce_inventory_demand_preprocessed.csv", index=False)
df.to_excel("C:/Users/manoj/Downloads/ecommerce_inventory_demand_preprocessed.xlsx", index=False)
print("\nPreprocessed data saved to CSV and Excel files.")
