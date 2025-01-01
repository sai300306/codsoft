#importing necessary dependencies
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import pandas as pd

#reading the data
titanic_data = pd.read_csv("C:/Users/Sai kumar/Downloads/archive (1)\Titanic-Dataset.csv")
print(titanic_data.head())
#checking for null values
#print(titanic_data.isnull().sum())
#dropping unnecessary columns
titanic_data_df = titanic_data.drop(['Cabin'],axis=1)

#filling null values
titanic_data_df.fillna({'Age':titanic_data_df['Age'].mean()},inplace=True)
titanic_data_df.fillna({'Embarked':titanic_data_df['Embarked'].mode()[0]},inplace=True)
#print(titanic_data_df.isnull().sum())

#replacing categorical data with numerical data
#print(titanic_data_df['Sex'].value_counts())
#print(titanic_data_df['Embarked'].value_counts())
pd.set_option('future.no_silent_downcasting', True)
titanic_data_df = titanic_data_df.replace({'Sex': {'male': 0, 'female': 1},'Embarked':{'S':0,'C':1,'Q':2}})
titanic_data_df = titanic_data_df.infer_objects(copy=False)
#print(titanic_data_df.head())

#splitting the training and testing data
X=titanic_data_df.drop(columns=['PassengerId','Ticket','Fare','Name','Survived'],axis=1)
Y=titanic_data_df['Survived']
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=2)
model = LogisticRegression()
model.fit(X_train,Y_train)


training_prediction = model.predict(X_train)
print("Training Accuracy score:",accuracy_score(Y_train,training_prediction)) 

testing_prediction = model.predict(X_test)
print("Testing Accuracy score:",accuracy_score(Y_test,testing_prediction)) 

print("Confusion Matrix:\n", confusion_matrix(Y_test, testing_prediction))
print("Classification Report:\n", classification_report(Y_test, testing_prediction))