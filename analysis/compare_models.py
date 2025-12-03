# Compare different models on their respective datasets and generate evaluation metrics

import pickle
import pandas as pd
import numpy as np
import os
from sklearn.metrics import (classification_report, accuracy_score, confusion_matrix, roc_curve, auc)
from analysis.preprocessing import load_dataset, clean_text


# All models and their datasets to evaluate
MODELS = [
    ("tfidf_logreg_spam", "data/spam.csv"),
    ("tfidf_logreg_spamham", "data/spam_ham_dataset.csv"),

    ("rf_spam", "data/spam.csv"),
    ("rf_spamham", "data/spam_ham_dataset.csv"),

    ("nb_spam", "data/spam.csv"),
    ("nb_spamham", "data/spam_ham_dataset.csv"),

    ("unified_logreg", "data/unified_spam_dataset.csv"),
    ("unified_rf", "data/unified_spam_dataset.csv"),
    ("unified_nb", "data/unified_spam_dataset.csv")
]

# Evaluate a single model on its dataset
def evaluate_model(model_name, data_path):
    print(f"\n=== Evaluating {model_name} on {data_path} ===")

    # Create output folder
    out_dir = f"analysis/results/{model_name}"
    os.makedirs(out_dir, exist_ok=True)

    # Load model + vectorizer
    model = pickle.load(open(f"models/{model_name}_model.pkl", "rb"))
    vectorizer = pickle.load(open(f"models/{model_name}_vectorizer.pkl", "rb"))

    # Load + preprocess dataset
    df = load_dataset(data_path)
    df["Message"] = df["Message"].apply(clean_text)

    X = vectorizer.transform(df["Message"])
    y = df["Category"].values

    # Predictions + probabilities
    preds = model.predict(X)
    probas = model.predict_proba(X)[:, list(model.classes_).index("Spam")]

    # Save prediction file
    pred_df = pd.DataFrame({
        "Message": df["Message"],
        "TrueLabel": y,
        "PredLabel": preds,
        "SpamProbability": probas
    })
    pred_df.to_csv(f"{out_dir}/predictions.csv", index=False)

    # Confusion matrix
    cm = confusion_matrix(y, preds, labels=["Not Spam", "Spam"])
    pd.DataFrame(cm, index=["True_NotSpam", "True_Spam"],
                 columns=["Pred_NotSpam", "Pred_Spam"]).to_csv(f"{out_dir}/confusion_matrix.csv")

    # ROC curve
    fpr, tpr, _ = roc_curve((y == "Spam").astype(int), probas)
    roc_auc = auc(fpr, tpr)
    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
    roc_df.to_csv(f"{out_dir}/roc_curve.csv", index=False)

    # Classification report
    report = classification_report(y, preds, output_dict=True)
    pd.DataFrame(report).to_csv(f"{out_dir}/classification_report.csv")

    acc = accuracy_score(y, preds)

    return {
        "model": model_name,
        "dataset": data_path,
        "accuracy": acc,
        "auc": roc_auc
    }


def main():
    os.makedirs("analysis/results", exist_ok=True)
    results = []

    for model_name, dataset_path in MODELS:
        try:
            metrics = evaluate_model(model_name, dataset_path)
            results.append(metrics)
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {e}")

    # Summary table
    df_results = pd.DataFrame(results)
    df_results.to_csv("analysis/results/summary_results.csv", index=False)

    print("\n=== Summary Table ===")
    print(df_results)


if __name__ == "__main__":
    main()
