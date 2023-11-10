import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('C://Users//Lenovo Thinkpad//Desktop//spam.csv', encoding='latin1')

# Drop non-required columns
data = data[['v1', 'v2']]


# Data preprocessing
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

data['v2'] = data['v2'].apply(clean_text)

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['v2'])
y = data['v1']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# Predict and evaluate the model
predictions = nb_classifier.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, predictions))

# Create a confusion matrix
conf_matrix = confusion_matrix(y_test, predictions)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# Count the occurrences of each label
label_counts = data['v1'].value_counts()

# Calculate percentages
total_messages = label_counts.sum()
spam_percentage = (label_counts['spam'] / total_messages) * 100
ham_percentage = (label_counts['ham'] / total_messages) * 100

# Plotting the bar graph
plt.figure(figsize=(6, 6))
label_counts.plot(kind='bar', color=['blue', 'green'])
plt.title('Distribution of Spam and Ham Messages')
plt.xlabel('Message Type')
plt.ylabel('Count')
plt.xticks(rotation=0)

# Annotate the percentages on the plot
plt.text(0, label_counts['ham'], f"Ham: {ham_percentage:.2f}%", ha='center', va='bottom')
plt.text(1, label_counts['spam'], f"Spam: {spam_percentage:.2f}%", ha='center', va='bottom')


plt.show()
