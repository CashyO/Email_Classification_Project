import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from analysis.preprocessing import load_dataset, clean_text

# Naive Bayes on Spam and SpamHam Datasets
def train_nb(dataset_path, model_name):

    # Load and preprocess data
    df = load_dataset(dataset_path)
    df["Message"] = df["Message"].apply(clean_text)

    # Feature extraction
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df["Message"])
    y = df["Category"]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train Model
    model = MultinomialNB()
    model.fit(X_train, y_train)

    print("\n=== Naive Bayes Results ===")
    print(classification_report(y_test, model.predict(X_test)))

    # Save the model and vectorizer to folder
    with open(f"models/{model_name}_model.pkl","wb") as f:
        pickle.dump(model, f)

    with open(f"models/{model_name}_vectorizer.pkl","wb") as f:
        pickle.dump(vectorizer, f)

if __name__ == "__main__":
    train_nb("data/spam.csv","nb_spam")
    train_nb("data/spam_ham_dataset.csv","nb_spamham")
