import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

# Load dataset
df = pd.read_csv("C:/Users/manoj/Downloads/ecommerce_inventory_demand_cleaned.csv")

# --- 1. Basic Overview ---
print("Shape of dataset:", df.shape)
print("/nData types:/n", df.dtypes)
print("/nSummary statistics:/n", df.describe())

# --- 2. Distribution of Numeric Columns ---
numeric_cols = df.select_dtypes(include='number').columns
numeric_cols = [col for col in numeric_cols if col not in ['discount_pct', 'is_promotion']]
for col in numeric_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f"Distribution of {col}")
    plt.show()

# --- 3. Boxplots for Outlier Visualization ---
for col in numeric_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

# --- 4. Correlation Heatmap ---
plt.figure(figsize=(10,6))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# --- 5. Category vs. Target Analysis ---
plt.figure(figsize=(8,5))
sns.barplot(x="product_category", y="units_sold", data=df, estimator=sum)
plt.title("Total Units Sold by Product Category")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(8,5))
sns.barplot(x="day_of_week", y="units_sold", data=df, estimator=sum)
plt.title("Units Sold by Day of Week")
plt.show()

plt.figure(figsize=(8,5))
sns.barplot(x="month", y="units_sold", data=df, estimator=sum)
plt.title("Units Sold by Month")
plt.show()

# --- 6. Promotion Impact ---
plt.figure(figsize=(6,4))
sns.boxplot(x="is_promotion", y="units_sold", data=df)
plt.title("Units Sold with vs. without Promotion")
plt.show()

# --- 7. Revenue Analysis ---
df['revenue'] = df['units_sold'] * df['unit_price']
plt.figure(figsize=(8,5))
sns.histplot(df['revenue'], bins=30, kde=True)
plt.title("Revenue Distribution")
plt.show()

# --- 8. Time Series Decomposition ---
# Ensure 'date' column is datetime and set as index
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Decompose the time series (adjust period based on your data frequency)
result = seasonal_decompose(df['units_sold'], model='additive', period=30)

# Decompose the time series
result = seasonal_decompose(df['units_sold'], model='additive', period=30)

# --- Custom Visualization ---
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

# Observed
axes[0].plot(result.observed, color="blue")
axes[0].set_title("Observed Units Sold", fontsize=12)

# Trend
axes[1].plot(result.trend, color="green")
axes[1].set_title("Trend Component (Long-term demand)", fontsize=12)

# Seasonal
axes[2].plot(result.seasonal, color="orange")
axes[2].set_title("Seasonal Component (Repeating cycles)", fontsize=12)

# Residual
axes[3].plot(result.resid, color="red")
axes[3].set_title("Residuals (Noise/Unexplained variation)", fontsize=12)

plt.suptitle("Time Series Decomposition of Units Sold", fontsize=14)
plt.tight_layout()
plt.show()
