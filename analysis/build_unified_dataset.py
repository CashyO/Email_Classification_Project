import pandas as pd
from analysis.preprocessing import clean_text

# Build a unified spam dataset from multiple sources
def build_unified_dataset():

    # Load dataset 1
    df1 = pd.read_csv("data/spam.csv", encoding="latin1")
    # Preprocess dataset 1
    df1 = df1[['v1', 'v2']]
    df1 = df1.rename(columns={'v1': 'Category', 'v2': 'Message'})
    df1['Category'] = df1['Category'].replace({'spam': 'Spam', 'ham': 'Not Spam'})

    # Load dataset 2 
    df2 = pd.read_csv("data/spam_ham_dataset.csv")
    # Preprocess dataset 2
    df2 = df2.rename(columns={'text': 'Message', 'label_num': 'LabelNum'})
    df2['Category'] = df2['LabelNum'].map({1: 'Spam', 0: 'Not Spam'})

    # Combine the datasets
    df = pd.concat([
        df1[['Message', 'Category']],
        df2[['Message', 'Category']]
    ], ignore_index=True)

    # Clean data - text normalization
    df['Message'] = df['Message'].astype(str).apply(clean_text)

    # Remove duplicates
    df = df.drop_duplicates()

    # Shuffle dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Creates the unified dataset
    df.to_csv("data/unified_spam_dataset.csv", index=False)
    print("Unified dataset created successfully!")
    print("Total rows:", len(df))

if __name__ == "__main__":
    build_unified_dataset()
