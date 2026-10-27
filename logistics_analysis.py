
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Generate logistics dataset
np.random.seed(42)
n = 500

data = {
    "Shipment_ID": range(1, n + 1),
    "Distance_km": np.random.randint(10, 1000, n),
    "Delivery_Time_Hours": np.random.randint(2, 72, n),
    "Shipment_Weight_kg": np.random.randint(1, 500, n),
    "Transport_Mode": np.random.choice(["Road", "Rail", "Air"], n),
    "Shipping_Cost": np.random.randint(500, 10000, n),
    "Delay_Hours": np.random.randint(0, 12, n),
    "Customer_Rating": np.random.randint(1, 6, n)
}

df = pd.DataFrame(data)

# Display dataset information
print("Dataset Shape:", df.shape)
print("\nDataset Information:")
df.info()

print("\nDescriptive Statistics:")
print(df.describe())

# Check missing values and duplicates
print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

# Correlation heatmap
numeric_cols = [
    "Distance_km",
    "Delivery_Time_Hours",
    "Shipment_Weight_kg",
    "Shipping_Cost",
    "Delay_Hours",
    "Customer_Rating"
]

plt.figure(figsize=(10, 6))
sns.heatmap(
    df[numeric_cols].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    vmin=-1,
    vmax=1
)
plt.title("Correlation Heatmap of Logistics Variables")
plt.tight_layout()
plt.show()

# Delivery time distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["Delivery_Time_Hours"], bins=20, kde=True)
plt.title("Distribution of Delivery Time")
plt.xlabel("Delivery Time (Hours)")
plt.tight_layout()
plt.show()

# Delivery time box plot
plt.figure(figsize=(8, 5))
sns.boxplot(x=df["Delivery_Time_Hours"])
plt.title("Box Plot of Delivery Time")
plt.tight_layout()
plt.show()

# Shipping cost distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["Shipping_Cost"], bins=20, kde=True)
plt.title("Distribution of Shipping Cost")
plt.xlabel("Shipping Cost")
plt.tight_layout()
plt.show()

# Distance versus shipping cost
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="Distance_km",
    y="Shipping_Cost"
)
plt.title("Distance vs Shipping Cost")
plt.tight_layout()
plt.show()

# Average shipping cost by transport mode
plt.figure(figsize=(8, 5))
sns.barplot(
    data=df,
    x="Transport_Mode",
    y="Shipping_Cost",
    errorbar=None
)
plt.title("Average Shipping Cost by Transport Mode")
plt.tight_layout()
plt.show()
