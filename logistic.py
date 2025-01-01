import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# Reading the CSV file
data = pd.read_csv('study_hours.csv')
# Splitting the dataset into independent (X) and dependent (y) variables
X = data[['Study Hours']] # Independent variable
y = data['Exam Result'] # Dependent variable
# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Creating and training the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)
# Making predictions on the test set
y_pred = model.predict(X_test)
# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
# Displaying results
print(f"Accuracy: {accuracy}")
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)
# Plotting the data points and decision boundary
plt.figure(figsize=(8, 6))
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.scatter(X_test, y_test, color='green', label='Testing Data', marker='x')
plt.xlabel('Study Hours')
plt.ylabel('Exam Result')
plt.title('Logistic Regression - Study Hours vs Exam Result')
# Generating the decision boundaryx_values = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_prob = model.predict_proba(x_values)[:, 1] # Probability for class 1
plt.plot(x_values, y_prob, color='red', label='Decision Boundary')
plt.legend()
plt.show()