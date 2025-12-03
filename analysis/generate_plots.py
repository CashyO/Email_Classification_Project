import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

RESULTS_DIR = "analysis/results"

# Generates plots for model comparison
def plot_accuracy_bar():
    df = pd.read_csv(f"{RESULTS_DIR}/summary_results.csv")
    plt.figure(figsize=(12,6))
    sns.barplot(x="model", y="accuracy", data=df)
    plt.xticks(rotation=45, ha='right')
    plt.title("Model Accuracy Comparison")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/accuracy_comparison.png")
    plt.close()

# Plots confusion matrices for each model
def plot_confusion_matrices():
    # Iterate through each model directory
    for model_name in os.listdir(RESULTS_DIR):
        if model_name == "summary_results.csv": continue
        cm_path = f"{RESULTS_DIR}/{model_name}/confusion_matrix.csv"
        if not os.path.exists(cm_path):
            continue

        # Load confusion matrix data
        cm = pd.read_csv(cm_path, index_col=0)
        plt.figure(figsize=(4,3))   
        sns.heatmap(cm, annot=True, cmap="Blues", fmt="d") 
        plt.title(f"Confusion Matrix - {model_name}") 
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/{model_name}/conf_matrix.png")
        plt.close()

# Plots ROC curves for each model
def plot_roc_curves():
    plt.figure(figsize=(8,6))
    # Iterate through each model directory
    for model_name in os.listdir(RESULTS_DIR):
        roc_path = f"{RESULTS_DIR}/{model_name}/roc_curve.csv"
        if not os.path.exists(roc_path):
            continue
        
        # Load ROC data
        roc = pd.read_csv(roc_path)
        plt.plot(roc["fpr"], roc["tpr"], label=model_name)

    plt.plot([0,1], [0,1], 'k--')
    plt.legend()
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/roc_curves.png")
    plt.close()

# Main function to generate all plots
def main():
    plot_accuracy_bar()
    plot_confusion_matrices()
    plot_roc_curves()
    print("Plots generated successfully!")

if __name__ == "__main__":
    main()
