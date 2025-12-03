CS 302 Cryptography & Network Security 

Spam Email Classification Project - Sentiment Analysis
@author Sebastian Osick

# Project Overview:
- The project classifies emails based on sentiment analysis where text input features are transformed into 
numeric vectors and then ran through machine learning algorithms to determine the class of an input feature. 
(spam = 1 & not spam = 0)
- This project intents to be an academic exercise to understand the relation between machine learning algorithms 
performance based on different feature sets. 
- The project also include a lightweight app, Streamlit, frontend user interface where users can upload email text for 
real-time classification of an email's class, spam or not spam. 

# Models Implemented
- Supervised Learning Algorithm: Logistic Regression 
- Probabilistic Learning Algorithm: Multinomal Naive Bayes
- Ensemble Tree Learning Algorithm: Random Forest Classifier 

# Data Sets Implemented 
 - All data was provided by Kaggle.com platform and community 
    - Spam Mails Dataset (spam_ham_dataset.csv)
    https://www.kaggle.com/datasets/venky73/spam-mails-dataset
    - SMS SPam Collection Dataset (spam.csv)
    https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
    - Unified Spam Dataset (unified_spam_dataset.csv)
    The concatenation of the two previous dataset into one larger set

# Project Architecture
Email_Classification_Project/
│
├── analysis/                       # Preprocessing & evaluation
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── build_unified_dataset.py
│   ├── compare_models.py
│   ├── generate_plots.py
│   ├── generate_pdf_report.py
│   ├── generate_report.py
│   └── results/
│
├── app/                            # Streamlit UI & prediction logic
│   ├── __init__.py
│   └── streamlit_app.py
│
├── data/                           # Raw email datasets 
│   ├── spam_ham_dataset.csv
│   ├── spam.csv
│   └── unified_spam_dataset.csv
│
├── models/                         # Serialized trained models (.pkl)
│   ├── ...
│
├── train/                          # Training scripts (NB, LR, RF)
│   ├── __init__.py
│   ├── train_logreg.py
│   ├── train_logreg_unified.py
│   ├── train_rf.py
│   ├── train_rf_unified.py
│   ├── train_nb_streamlit.py
│   └── train_rf_unified.py
│
├── requirements.txt                # Git & Github
├── README.md
├── __init__.py
└── .gitignore


# Create a virtual environment
python -m venv venv
# Activate the virtual environment (windows)
venv\Scripts\activate

# To install the required packages
pip install -r requirements.txt

# Once your IDE is configured, a series of commands to run the project
# To load the datasets
python -m analysis.build_unified_dataset
python -m analysis.preprocessing

# To train the models 
python -m train.train_logreg
python -m train.train_rf
python -m train.train_nb_streamlit

python -m train.train_logreg_unified
python -m train.train_rf_unified
python -m train.train_nb_unified

# To compare the models
python -m analysis.compare_models

# To create the visualizations
python -m analysis.generate_plots
python -m analysis.generate_report

# To run the streamlit app
streamlit run app/streamlit_app.py

# Test inputs 
Spam Example
    "Congratulations! You've won a $1,000 Walmart gift card. Click here to claim your prize."
Not Spam Example
    "Hi team, please find attached the minutes from today's meeting. Let me know if you have any questions."

