import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

data = pd.read_csv('/path/to/smsspamcollection/SMSSpamCollection', sep='\t' , names = ['label', 'text'])

# Display the first few rows of the DataFrame
print(data.head())
# Display the last few rows of the DataFrame
print(data.tail())
# Display the shape of the DataFrame
print(data.shape)
# Display the data types of the columns
print(data.dtypes)
# Display the number of missing values in each column
print(data.isnull().sum())
# Display the unique values in the 'label' column
print(data.label.unique())
# Display the value counts of the 'label' column
print(data.label.value_counts())


# Function to clean text
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# Apply the clean_text function to the 'text' column
#data['text'] = data['text'].apply(clean_text)
# Apply cleaning
data['clean_text'] = data['text'].apply(clean_text)

print(data[['text', 'clean_text']].head())
# Display the first few rows of the cleaned DataFrame
#print(data.head())
#print(data.tail())


# Initialize TF-IDF
vectorizer = TfidfVectorizer()

# Transform the cleaned text
X = vectorizer.fit_transform(data['clean_text'])

# Labels
y = data['label'].map({'ham': 0, 'spam': 1})  # convert labels to 0 and 1

print(X.shape)
print(y.shape)
print(data.dtypes)
print(data.isnull().sum())


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape)


# Initialize model
model = LogisticRegression()

# Train
model.fit(X_train, y_train)


# Predict on test set
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Detailed report
print(classification_report(y_test, y_pred))


# Save the model
with open('app/spam_classifier_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save the vectorizer
with open('app/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

