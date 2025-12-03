import pandas as pd
import os

# Pull the analysis results directory for report generation
RESULTS_DIR = "analysis/results"

# Generates a markdown report summarizing model results
def generate_report():

    # Load summary results
    df = pd.read_csv(f"{RESULTS_DIR}/summary_results.csv")

    md = "# Model Comparison Report\n\n"
    md += "Generated automatically.\n\n"

    md += "## Summary Table\n\n"
    md += df.to_markdown(index=False)
    md += "\n\n"

    # Accuracy plot
    md += "## Accuracy Comparison\n"
    md += "![Accuracy](accuracy_comparison.png)\n\n"

    # ROC plot
    md += "## ROC Curves\n"
    md += "![ROC](roc_curves.png)\n\n"

    # Confusion Matrices Per Model
    md += "## Confusion Matrices Per Model\n\n"
    for model in df["model"]:
        cm_path = f"{RESULTS_DIR}/{model}/conf_matrix.png"
        if os.path.exists(cm_path):
            md += f"### {model}\n"
            md += f"![]({model}/conf_matrix.png)\n\n"

    # Save report
    with open(f"{RESULTS_DIR}/model_report.md", "w") as f:
        f.write(md)

    print("Markdown report generated!")

if __name__ == "__main__":
    generate_report()

