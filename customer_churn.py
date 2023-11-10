# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# Load the dataset
df = pd.read_csv('C://Users//Lenovo Thinkpad//Desktop//Churn_Modelling.csv')

# Display the first few rows of the dataset to understand its structure
print(df.head())

# Drop unnecessary columns (e.g., customer ID, irrelevant features)
# Replace this with your actual column names
features_to_drop = ['RowNumber', 'CustomerId', 'Surname']
df = df.drop(features_to_drop, axis=1)

# Check for missing values and handle them if needed
print(df.isnull().sum())

# Handle categorical variables using Label Encoding
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df['Geography'] = le.fit_transform(df['Geography'])

# Define features (X) and target variable (y)
X = df.drop('Exited', axis=1)
y = df['Exited']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train a Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Print the evaluation metrics
print(f'Accuracy: {accuracy:.2f}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{classification_rep}')

# Plot a bar chart to visualize the distribution of classes
plt.figure(figsize=(8, 6))
sns.countplot(x=y_test, palette='viridis')
plt.title('Distribution of Classes in the Test Set')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

# Plot a bar chart to visualize the percentage of correct classifications
plt.figure(figsize=(8, 6))
sns.countplot(x=y_test, hue=y_pred, palette='viridis')
plt.title('Correct Classification Percentage')
plt.xlabel('Actual Class')
plt.ylabel('Count')
plt.legend(title='Predicted Class')
plt.show()
