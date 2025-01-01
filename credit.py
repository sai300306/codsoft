import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Load dataset
credit_dataset = pd.read_csv("C:/Users/Sai kumar/Downloads/archive (3)/creditcard.csv")

# Analyze class distribution and sample the dataset
legit = credit_dataset[credit_dataset.Class == 0]
fraud = credit_dataset[credit_dataset.Class == 1]
legit_sample = legit.sample(n=492)
new_dataset = pd.concat([legit_sample, fraud], axis=0)

# Split features and target variable
X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define classifiers
models = {
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Logistic Regression': LogisticRegression(),
    'Support Vector Machine': SVC(kernel='linear'),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}

# Evaluate each model
performance = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, Y_train)
    Y_pred_train = model.predict(X_train)
    Y_pred_test = model.predict(X_test)

    train_accuracy = accuracy_score(Y_train, Y_pred_train)
    test_accuracy = accuracy_score(Y_test, Y_pred_test)
    performance[name] = test_accuracy

    print(f"Train Accuracy: {train_accuracy:.2f}")
    print(f"Test Accuracy: {test_accuracy:.2f}")
    print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_pred_test))
    print("Classification Report:\n", classification_report(Y_test, Y_pred_test))

# Compare performance
plt.figure(figsize=(10, 5))
sns.barplot(x=list(performance.keys()), y=list(performance.values()), hue=None)
plt.title("Model Performance Comparison")
plt.ylabel("Accuracy")
plt.xlabel("Model")
plt.show()