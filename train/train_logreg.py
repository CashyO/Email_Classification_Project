import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from analysis.preprocessing import load_dataset, clean_text

# Logistic Regression on Spam and SpamHam Datasets
def train_model(dataset_path, model_name):

    # Load and preprocess data
    df = load_dataset(dataset_path)
    df["Message"] = df["Message"].apply(clean_text)

    # Feature extraction
    X = df["Message"]
    y = df["Category"]

    # Convert text into Vector form
    vectorizer = TfidfVectorizer(stop_words='english')
    X_vec = vectorizer.fit_transform(X)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2)

    # Train Model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    print("\n=== Logistic Regression Results ===")
    print(classification_report(y_test, model.predict(X_test)))

    # Save the model and vectorizer to folder
    with open(f"models/{model_name}_model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open(f"models/{model_name}_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

if __name__ == "__main__":
    train_model("data/spam.csv", "tfidf_logreg_spam")
    train_model("data/spam_ham_dataset.csv", "tfidf_logreg_spamham")
