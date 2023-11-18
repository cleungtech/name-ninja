import mysql.connector
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Database connection
db = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="",
    database="name_ninja"
)

# SQL query to retrieve data
query = "SELECT Full_Name, Prefix, Suffix, First_Name, Middle_Name, Last_Name FROM name_ninja.training_data"

# Execute query and fetch data
cursor = db.cursor()
cursor.execute(query)
result = cursor.fetchall()

# Data preparation
data = []
labels = []
for row in result:
    full_name, prefix, suffix, first_name, middle_name, last_name = row
    for word in full_name.split():
        if word == prefix:
            labels.append('prefix')
        elif word == first_name:
            labels.append('first')
        elif word == middle_name:
            labels.append('middle')
        elif word == last_name:
            labels.append('last')
        elif word == suffix:
            labels.append('suffix')
        else:
            labels.append('unknown')
        data.append(word)

# Convert words to numerical data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data)

# Convert labels to numerical data
le = LabelEncoder()
y = le.fit_transform(labels)

# Splitting the dataset for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
def parse_name(model, vectorizer, label_encoder, name):
    # Split the name into words
    words = name.split()

    # Transform the words using the vectorizer
    X = vectorizer.transform(words)

    # Predict the category of each word
    predictions = model.predict(X)

    # Decode the predicted categories to their original labels
    predicted_labels = label_encoder.inverse_transform(predictions)

    # Combine words with their predicted labels
    parsed_name = dict(zip(words, predicted_labels))

    return parsed_name

# Example usage
trained_model = clf
trained_vectorizer = vectorizer
trained_label_encoder = le

# Parse a new name
new_name = "Dr. Emily Jane Watson"
parsed_name = parse_name(trained_model, trained_vectorizer, trained_label_encoder, new_name)
print(parsed_name)