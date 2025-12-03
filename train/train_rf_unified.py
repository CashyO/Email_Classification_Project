import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from analysis.preprocessing import clean_text

# Random Forest on Unified Spam Dataset
def train_unified_rf():

    # Load and preprocess data
    df = pd.read_csv("data/unified_spam_dataset.csv")
    df['Message'] = df['Message'].apply(clean_text)

    # Feature extraction
    X = df['Message']
    y = df['Category']

    # Convert text into Vector form
    vectorizer = CountVectorizer(stop_words="english")
    X_vec = vectorizer.fit_transform(X)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

    # Train Model
    model = RandomForestClassifier(n_estimators=300, n_jobs=-1)
    model.fit(X_train, y_train)

    print("\n=== Unified Random Forest ===")
    print(classification_report(y_test, model.predict(X_test)))

    # Save the model and vectorizer to folder
    pickle.dump(model, open("models/unified_rf_model.pkl", "wb"))
    pickle.dump(vectorizer, open("models/unified_rf_vectorizer.pkl", "wb"))

if __name__ == "__main__":
    train_unified_rf()

