import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv("C:/Users/Sai kumar/Downloads/archive (2)/IRIS.csv")
print(df.head())

# Define feature set (X) and target variable (Y)
X = df[['sepal_length', 'sepal_width']]  # Use only sepal_length and sepal_width
Y = df['species'] 

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=101)

# Create and train the logistic regression model
model = LogisticRegression(max_iter=200)  # Increase max_iter in case of convergence issues
model.fit(X_train, Y_train)

# Get training and testing accuracy
training_prediction = model.predict(X_train)
print("Training Accuracy:", accuracy_score(Y_train, training_prediction))

testing_prediction = model.predict(X_test)
print("Testing Accuracy:", accuracy_score(Y_test, testing_prediction))

# Create meshgrid for the contour plot (2D grid for 'sepal_length' and 'sepal_width')
sepal_length = np.linspace(df['sepal_length'].min(), df['sepal_length'].max(), 100)
sepal_width = np.linspace(df['sepal_width'].min(), df['sepal_width'].max(), 100)

grid_sepal_length, grid_sepal_width = np.meshgrid(sepal_length, sepal_width)

# Create grid of features to predict on
grid_input = pd.DataFrame(
    np.c_[grid_sepal_length.ravel(), grid_sepal_width.ravel()],
    columns=['sepal_length', 'sepal_width']
)

# Predict using the trained model
grid_predictions = model.predict(grid_input)

# Convert categorical labels to numerical values for plotting (e.g., 0, 1, 2 for the species)
label_map = {label: idx for idx, label in enumerate(np.unique(Y))}
grid_predictions_numerical = np.array([label_map[label] for label in grid_predictions])

# Reshape grid_predictions_numerical to match the grid shape
grid_predictions_numerical = grid_predictions_numerical.reshape(grid_sepal_length.shape)

# Plot the contour and scatter plot
plt.figure(figsize=(10, 6))
cp = plt.contourf(grid_sepal_length, grid_sepal_width, grid_predictions_numerical, alpha=0.3, cmap='coolwarm')

# Scatter plot for actual data points
plt.scatter(X['sepal_length'], X['sepal_width'], c=Y.astype('category').cat.codes, cmap='coolwarm', edgecolor='k', label='Actual Data')

# Plot details
plt.title('Logistic Regression Decision Boundary (Sepal Length vs Sepal Width)', fontsize=14)
plt.xlabel('Sepal Length', fontsize=12)
plt.ylabel('Sepal Width', fontsize=12)
plt.colorbar(cp, label='Predicted Class')
plt.legend()
plt.show()
