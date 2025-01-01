import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data from the CSV file
file_path = r"C:\Users\Sai kumar\Downloads\advertising.csv"  # Use raw string to handle the backslashes
df = pd.read_csv(file_path)

# Basic Descriptive Statistics
print("Descriptive Statistics:")
print(df.describe())

# Correlation Matrix
print("\nCorrelation Matrix:")
print(df.corr())

# Visualizing the data - Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# Scatter plots to visualize relationships between the predictors and sales
sns.pairplot(df, hue='Sales', palette='viridis')
plt.show()
