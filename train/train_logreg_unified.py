import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from analysis.preprocessing import clean_text

# Logistic Regression on Unified Spam Dataset
def train_unified_logreg():

    # Load and preprocess data
    df = pd.read_csv("data/unified_spam_dataset.csv")
    df['Message'] = df['Message'].apply(clean_text)

    # Feature extraction  
    X = df['Message']
    y = df['Category']

    # Convert text into Vector form 
    vectorizer = TfidfVectorizer(stop_words="english")
    X_vec = vectorizer.fit_transform(X)

    # Split data into train and test sets 
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

    # Train Model 
    model = LogisticRegression(max_iter=300)
    model.fit(X_train, y_train)

    print("\n=== Unified Logistic Regression ===")
    print(classification_report(y_test, model.predict(X_test)))

    # Save the model and vectorizer to folder 
    pickle.dump(model, open("models/unified_logreg_model.pkl", "wb"))
    pickle.dump(vectorizer, open("models/unified_logreg_vectorizer.pkl", "wb"))

if __name__ == "__main__":
    train_unified_logreg()
