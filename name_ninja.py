import mysql.connector
import os
import joblib
from db_config import get_db_connection
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class NameParserModel:
    def __init__(self, db_config, model_filename='name_ninja_model.joblib'):
        self.db_config = db_config
        self.model_filename = model_filename
        self.vectorizer = CountVectorizer()
        self.label_encoder = LabelEncoder()
        self.clf = None

    def _connect_db(self):
        return self.db_config

    def load_data(self):
        db = self._connect_db()
        query = "SELECT Full_Name, Prefix, Suffix, First_Name, Middle_Name, Last_Name FROM training_data"
        cursor = db.cursor()
        cursor.execute(query)
        return cursor.fetchall()

    def prepare_data(self, data):
        feature_data = []
        labels = []
        for row in data:
            full_name, prefix, suffix, first_name, middle_name, last_name = row
            for word in full_name.split():
                label = 'unknown'  # Default label
                if word == prefix:
                    label = 'prefix'
                elif word == suffix:
                    label = 'suffix'
                elif word == first_name:
                    label = 'first'
                elif word == middle_name:
                    label = 'middle'
                elif word == last_name:
                    label = 'last'
                labels.append(label)
                feature_data.append(word)

        X = self.vectorizer.fit_transform(feature_data)
        y = self.label_encoder.fit_transform(labels)
        return X, y

    def train_model(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        self.clf = DecisionTreeClassifier()
        self.clf.fit(X_train, y_train)
        y_pred = self.clf.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))

    def save_model(self):
        if self.clf:
            # Save classifier, vectorizer, and label encoder
            joblib.dump((self.clf, self.vectorizer, self.label_encoder), self.model_filename)

    def load_model(self):
        if os.path.exists(self.model_filename):
            # Load classifier, vectorizer, and label encoder
            self.clf, self.vectorizer, self.label_encoder = joblib.load(self.model_filename)

    def predict(self, name):
        if not self.clf:
            raise Exception("Model not loaded or trained.")
        name_tokens = name.split()
        transformed_name = self.vectorizer.transform(name_tokens)
        prediction = self.clf.predict(transformed_name)
        return self.label_encoder.inverse_transform(prediction)


if __name__ == '__main__':
    db_config = get_db_connection()
    model = NameParserModel(db_config)
    model.load_model()

    if not model.clf:
        data = model.load_data()
        X, y = model.prepare_data(data)
        model.train_model(X, y)
        model.save_model()

    # Prediction example
    example_name = "Dr. John M. Doe Jr."
    predicted_label = model.predict(example_name)
    print(predicted_label)
