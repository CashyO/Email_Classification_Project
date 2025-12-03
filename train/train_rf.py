import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from analysis.preprocessing import load_dataset, clean_text

# Random Forest on Spam and SpamHam Datasets
def train_rf(dataset_path, model_name):

    # Load and preprocess data
    df = load_dataset(dataset_path)
    df["Message"] = df["Message"].apply(clean_text)

    # Feature extraction
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df["Message"])
    y = df["Category"]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train Model
    clf = RandomForestClassifier(n_jobs=-1)
    clf.fit(X_train, y_train)

    print("\n=== Random Forest Results ===")
    print(classification_report(y_test, clf.predict(X_test)))

    # Save the model and vectorizer to folder
    with open(f"models/{model_name}_model.pkl","wb") as f:
        pickle.dump(clf, f)

    with open(f"models/{model_name}_vectorizer.pkl","wb") as f:
        pickle.dump(vectorizer, f)

if __name__ == "__main__":
    train_rf("data/spam.csv","rf_spam")
    train_rf("data/spam_ham_dataset.csv","rf_spamham")
