# Processes and normalizes text datasets 

import string
import numpy as np
import pandas as pd

# Load dataset with multiple format (troubleshooting)
def load_dataset(path):

    # Load CSV 
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(path, encoding="latin1")
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="windows-1252")

    # Identify dataset type and standardize columns
    # Case 1: spam.csv format
    if "v1" in df.columns and "v2" in df.columns:
        df = df[["v1", "v2"]]
        df = df.rename(columns={"v1": "Category", "v2": "Message"})
        df["Category"] = df["Category"].replace({"spam": "Spam", "ham": "Not Spam"})

    # Case 2: spam_ham_dataset.csv format
    elif "label_num" in df.columns and "text" in df.columns:
        df = df.rename(columns={"text": "Message", "label_num": "LabelNum"})
        df["Category"] = df["LabelNum"].map({1: "Spam", 0: "Not Spam"})

    # Case 3: unified_spam_dataset.csv format 
    elif "Message" in df.columns and "Category" in df.columns:
        df = df[["Message", "Category"]]

    else:
        raise ValueError(f"Unrecognized dataset format for file: {path}")

    # Standardize labels across all datasets 
    df["Category"] = df["Category"].replace({
        1: "Spam",
        0: "Not Spam",
        "spam": "Spam",
        "ham": "Not Spam",
        "nan": "Not Spam",
        None: "Not Spam",
    })

    # Drop rows with missing Message or Category labels
    df = df.dropna(subset=["Message", "Category"])

    # Ensure Message is always a string 
    df["Message"] = df["Message"].astype(str)

    # Remove blank messages
    df = df[df["Message"].str.strip() != ""]
    df = df[df["Message"].str.lower() != "nan"]

    # Return the cleaned DataFrame
    return df

# Text normalization
def clean_text(text):

    # Handle missing or bad types (floats, NaN, None)
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Return cleaned text
    return text

