import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the training dataset
train_data = pd.read_csv('C://Users//Lenovo Thinkpad//Desktop//fraudTrain.csv')

# Display the first few rows of the dataset to understand its structure
print(train_data.head())

# Define features (X) and target variable (y) for training data
X_train = train_data.drop(['is_fraud', 'trans_date_trans_time', 'cc_num', 'merchant', 'category', 'gender', 'first', 'last', 'street', 'city', 'state', 'zip', 'job', 'dob', 'trans_num'], axis=1)
y_train = train_data['is_fraud']

# Standardize the feature values for training data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Initialize and train a Logistic Regression model
logreg_model = LogisticRegression(random_state=42)
logreg_model.fit(X_train, y_train)

# Load the test dataset
test_data = pd.read_csv('C://Users//Lenovo Thinkpad//Desktop//fraudTest.csv')

# Display the first few rows of the test dataset
print(test_data.head())

# Define features (X) and target variable (y) for test data
X_test = test_data.drop(['is_fraud', 'trans_date_trans_time', 'cc_num', 'merchant', 'category', 'gender', 'first', 'last', 'street', 'city', 'state', 'zip', 'job', 'dob', 'trans_num'], axis=1)
y_test = test_data['is_fraud']

# Standardize the feature values for test data using the same scaler as for training data
X_test = scaler.transform(X_test)

# Make predictions on the test set
y_pred = logreg_model.predict(X_test)

# Evaluate the model on the test set
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Print the evaluation metrics for the test set
print(f'Test Set Accuracy: {accuracy:.2f}')
print(f'Test Set Confusion Matrix:\n{conf_matrix}')
print(f'Test Set Classification Report:\n{classification_rep}')

# Visualize the distribution of fraud and non-fraud instances
plt.figure(figsize=(8, 6))
sns.countplot(x=y_test, hue=y_pred, palette='viridis', legend=False)
plt.title('Distribution of Classes in the Test Set')
plt.xlabel('Class (0: Not Fraud, 1: Fraud)')
plt.ylabel('Count')
plt.show()
