import sys
import os

# To see the project root as a package
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from analysis.preprocessing import clean_text


# Implement Frontend via streamlit 
import streamlit as st
import pickle
import string
from analysis.preprocessing import clean_text

# Available Models to Test
MODELS = {
    "Logistic Regression (spam.csv)": ("tfidf_logreg_spam_model.pkl", "tfidf_logreg_spam_vectorizer.pkl"),
    "Logistic Regression (spam_ham_dataset.csv)": ("tfidf_logreg_spamham_model.pkl", "tfidf_logreg_spamham_vectorizer.pkl"),
    "Unified Logistic Regression (ALL DATA)": ("unified_logreg_model.pkl", "unified_logreg_vectorizer.pkl"),

    "Random Forest (spam.csv)": ("rf_spam_model.pkl", "rf_spam_vectorizer.pkl"),
    "Random Forest (spam_ham_dataset.csv)": ("rf_spamham_model.pkl", "rf_spamham_vectorizer.pkl"),
    "Unified Random Forest (ALL DATA)": ("unified_rf_model.pkl","unified_rf_vectorizer.pkl"),

    "Naive Bayes (spam.csv)": ("nb_spam_model.pkl", "nb_spam_vectorizer.pkl"),
    "Naive Bayes (spam_ham_dataset.csv)": ("nb_spamham_model.pkl", "nb_spamham_vectorizer.pkl"),
    "Unified Naive Bayes (ALL DATA)": ("unified_nb_model.pkl", "unified_nb_vectorizer.pkl"),

}

# Streamlit UI
st.set_page_config(page_title="Unified Spam Detection System", page_icon="📧")

st.title("📧 Unified Email Spam Detection System")
st.write("Choose a model and dataset to classify your message.")

st.markdown("---")

# Drop down model selection menu 
model_choice = st.selectbox("Choose Model:", list(MODELS.keys()))

# Load model + vectorizer
model_path, vectorizer_path = MODELS[model_choice]

model = pickle.load(open(f"models/{model_path}", "rb"))
vectorizer = pickle.load(open(f"models/{vectorizer_path}", "rb"))

# User input 
input_message = st.text_area("✉️ Enter your email/message:", height=150)

if st.button("🔍 Predict"):
    if input_message.strip() == "":
        st.warning("Please enter a message before predicting.")
    else:
        cleaned = clean_text(input_message)
        features = vectorizer.transform([cleaned]).toarray()

        prediction = model.predict(features)[0]
        prob = model.predict_proba(features)[0]

        # Display result
        if prediction == "Spam":
            st.error("🚫 **SPAM DETECTED**")
            idx = list(model.classes_).index("Spam")
            st.write("Confidence:", f"{prob[idx]:.4f}")
        else:
            st.success("✅ **NOT SPAM**")
            idx = list(model.classes_).index("Not Spam")
            st.write("Confidence:", f"{prob[idx]:.4f}")

        st.markdown("---")
        st.subheader("Preprocessed Message")
        st.code(cleaned)
