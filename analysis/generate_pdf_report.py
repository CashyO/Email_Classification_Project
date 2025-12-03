from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
import pandas as pd
import os

RESULTS_DIR = "analysis/results"

# Generates a PDF report summarizing model results
def generate_pdf_report():

    pdf_path = f"{RESULTS_DIR}/model_report.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)

    styles = getSampleStyleSheet()
    story = []

    # Title Page
    story.append(Paragraph("<b>Machine Learning Spam Detection Report</b>", styles["Title"]))
    story.append(Spacer(1, 20))

    story.append(Paragraph("Automatically generated evaluation report.", styles["BodyText"]))
    story.append(Spacer(1, 20))

    story.append(PageBreak())

    # Summary Table
    df = pd.read_csv(f"{RESULTS_DIR}/summary_results.csv")

    story.append(Paragraph("<b>Model Performance Summary</b>", styles["Heading2"]))
    story.append(Spacer(1, 10))

    table_html = df.to_string(index=False).replace("\n", "<br/>")
    story.append(Paragraph(f"<font face='Courier'>{table_html}</font>", styles["BodyText"]))
    story.append(Spacer(1, 20))

    # Accuracy Plot
    acc_path = f"{RESULTS_DIR}/accuracy_comparison.png"
    if os.path.exists(acc_path):
        story.append(Paragraph("<b>Accuracy Comparison Chart</b>", styles["Heading2"]))
        story.append(Image(acc_path, width=500, height=300))
        story.append(Spacer(1, 20))

    # ROC Curves
    roc_path = f"{RESULTS_DIR}/roc_curves.png"
    if os.path.exists(roc_path):
        story.append(Paragraph("<b>ROC Curve Comparison</b>", styles["Heading2"]))
        story.append(Image(roc_path, width=500, height=300))
        story.append(Spacer(1, 20))

    story.append(PageBreak())

    # Confusion Matrices Per Model
    story.append(Paragraph("<b>Confusion Matrices (Per Model)</b>", styles["Heading2"]))
    story.append(Spacer(1, 20))

    for model in df["model"]:
        model_cm_path = f"{RESULTS_DIR}/{model}/conf_matrix.png"

        if os.path.exists(model_cm_path):
            story.append(Paragraph(f"<b>{model}</b>", styles["Heading3"]))
            story.append(Spacer(1, 10))

            story.append(Image(model_cm_path, width=350, height=350))
            story.append(Spacer(1, 20))

    # Create PDF
    doc.build(story)

    print(f"PDF Report generated: {pdf_path}")


if __name__ == "__main__":
    generate_pdf_report()
