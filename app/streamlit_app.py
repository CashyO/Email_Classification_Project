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

        # Explain prediction via feature weights
        st.markdown("---")
        st.subheader("Top Contributing Words (Spam Triggers)")

        feature_names = (
            vectorizer.get_feature_names_out()
            if hasattr(vectorizer, "get_feature_names_out")
            else vectorizer.get_feature_names()
        )

        classes = list(model.classes_)
        spam_idx = classes.index("Spam") if "Spam" in classes else 1

        weights = None
        if hasattr(model, "coef_"):
            if len(classes) == 2 and model.coef_.shape[0] == 1:
                weights = model.coef_[0] if spam_idx == 1 else -model.coef_[0]
            else:
                weights = model.coef_[spam_idx]
        elif hasattr(model, "feature_log_prob_"):
            weights = model.feature_log_prob_[spam_idx]
        elif hasattr(model, "feature_importances_"):
            weights = model.feature_importances_

        if weights is not None and feature_names is not None:
            non_zero_indices = features[0].nonzero()[0]
            if len(non_zero_indices) > 0:
                word_weights = [(feature_names[i], float(weights[i])) for i in non_zero_indices]
                word_weights.sort(key=lambda x: x[1], reverse=True)
                top_words = word_weights[:10]
                explanation_data = [{"Word": word, "Weight": round(weight, 4)} for word, weight in top_words]
                st.table(explanation_data)
            else:
                st.info("No matching vocabulary words found in the message.")
        else:
            st.info("Feature weights are not available for this model.")

        st.markdown("---")
        st.subheader("Preprocessed Message")
        st.code(cleaned)
