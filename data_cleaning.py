import pandas as pd 
import openpyxl

df = pd.read_csv("C:/Users/manoj/Downloads/ecommerce_inventory_demand (2).csv")

# Null values check
null_values = df.isnull().sum()
print("Null values in each column:")
print(null_values)

# Outlier detection
outlier_counts = {}
for col in df.select_dtypes(include="number").columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
    outlier_counts[col] = len(outliers)

# Show only outlier counts
print("/nOutlier counts per column:")
for col, count in outlier_counts.items():
    if count > 0:
        print(f"{col}: {count}")

# --- Continuation: Impute Nulls and Cap Outliers ---

# 1. Impute nulls in 'units_sold' (only column with missing values)
df['units_sold'] = df['units_sold'].fillna(df['units_sold'].median())

# Final check
print("\nNulls after imputation:")
print(df['units_sold'].isnull().sum())

# --- Store Original Bounds for Each Column ---
bounds = {}
for col in df.select_dtypes(include="number").columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    bounds[col] = (lower_bound, upper_bound)
# --- Define continuous numeric columns only ---
continuous_cols = ['units_sold', 'unit_price', 'stock_on_hand', 'reorder_point', 'supplier_lead_days']
# --- Cap Outliers Using Stored Bounds ---
for col in continuous_cols:
    lower_bound, upper_bound = bounds[col]
    df[col] = df[col].apply(lambda x: 
                            lower_bound if x < lower_bound else 
                            upper_bound if x > upper_bound else x)

# --- Outlier Check Using Original Bounds ---
print("\nOutlier counts per column AFTER capping (using original bounds):")
for col in continuous_cols:
    lower_bound, upper_bound = bounds[col]
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
    if len(outliers) > 0:
        print(f"{col}: {len(outliers)}")
    else:
        print(f"{col}: No outliers")
        
df.to_csv(r"C:\Users\manoj\Downloads\ecommerce_inventory_demand_cleaned.csv", index=False)
df.to_excel(r"C:\Users\manoj\Downloads\ecommerce_inventory_demand_cleaned.xlsx", index=False)
print("\nCleaned data saved to CSV and Excel files.")
