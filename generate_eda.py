import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('Car details v3.csv')

# Clean attributes & extract numeric features
df['brand'] = df['name'].apply(lambda x: str(x).split()[0])
df['mileage_num'] = df['mileage'].str.extract(r'([\d.]+)').astype(float)
df['engine_num'] = df['engine'].str.extract(r'([\d.]+)').astype(float)
df['max_power_num'] = pd.to_numeric(df['max_power'].str.extract(r'([\d.]+)')[0], errors='coerce')
df['car_age'] = 2026 - df['year']

df_clean = df.drop(columns=['name', 'year', 'mileage', 'engine', 'max_power', 'torque'])

# Plotting EDA Dashboard
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

sns.histplot(df_clean['selling_price'], kde=True, ax=axes[0, 0], color='skyblue')
axes[0, 0].set_title('1. Target Distribution (Selling Price)')

sns.scatterplot(data=df_clean, x='max_power_num', y='selling_price', hue='fuel', alpha=0.7, ax=axes[0, 1])
axes[0, 1].set_title('2. Max Power vs Selling Price')

sns.scatterplot(data=df_clean, x='car_age', y='selling_price', color='teal', alpha=0.6, ax=axes[0, 2])
axes[0, 2].set_title('3. Car Age vs Selling Price')

sns.boxplot(data=df_clean, x='fuel', y='selling_price', ax=axes[1, 0], palette='Set2')
axes[1, 0].set_title('4. Fuel Type vs Selling Price')

corr = df_clean[['selling_price', 'km_driven', 'mileage_num', 'engine_num', 'max_power_num', 'car_age']].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[1, 1], cbar=False)
axes[1, 1].set_title('5. Feature Correlation Heatmap')

axes[1, 2].axis('off')
plt.tight_layout()
plt.savefig('eda_visualizations.png', dpi=300)
print("EDA charts successfully generated and saved as 'eda_visualizations.png'.")