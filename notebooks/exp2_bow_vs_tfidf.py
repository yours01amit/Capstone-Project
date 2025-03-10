import os
import re
import setuptools
import pandas as pd
import string

pd.set_option('future.no_silent_downcasting', True)

import numpy as np
import mlflow
import mlflow.sklearn
import dagshub

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import scipy.sparse

import warnings
warnings.simplefilter('ignore', UserWarning)
warnings.filterwarnings('ignore')


# ------------------------------ CONFIGURATION --------------------------

CONFIG = {
    "data_path": "notebooks/IMDB.csv",
    "test_size": 0.2,
    "mlflow_tracking_uri": "https://dagshub.com/ak0614653/Capstone-Project.mlflow",
    "dagshub_repo_owner": "ak0614653",
    "dagshub_repo_name": "Capstone-Project",
    "experiment_name": "Bow vs Tfidf"
}


#--------------------------- SETUP MLFLOW & DAGSHUB -----------------------------

mlflow.set_tracking_uri(CONFIG["mlflow_tracking_uri"])
dagshub.init(
    repo_owner=CONFIG["dagshub_repo_owner"], 
    repo_name=CONFIG["dagshub_repo_name"], 
    mlflow=True
)
mlflow.set_experiment(CONFIG["experiment_name"])


# ---------------------------- TEXT PREPROCESSING ------------------------------

def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])


def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    return " ".join([word for word in text.split() if word not in stop_words])


def removing_numbers(text):
    return ''.join([char for char in text if not char.isdigit()])


def lower_case(text):
    return text.lower()


def removing_punctuations(text):
    return re.sub(f"[{re.escape(string.punctuation)}]", ' ', text)


def removing_urls(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text)


def normalize_text(df):
    try:
        df['review'] = df['review'].apply(lower_case)
        df['review'] = df['review'].apply(remove_stop_words)
        df['review'] = df['review'].apply(removing_numbers)
        df['review'] = df['review'].apply(removing_punctuations)
        df['review'] = df['review'].apply(removing_urls)
        df['review'] = df['review'].apply(lemmatization)
        return df
    except Exception as e:
        print(f"Error during text normalization: {e}")
        raise


# ----------------------- LOAD AND PREPROCESS THE DATA ------------------------------

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df = normalize_text(df)
        df = df[df['sentiment'].isin(['positive', 'negative'])]
        df['sentiment'] = df['sentiment'].replace({'negative': 0, 'positive': 1}).infer_objects(copy=False)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise


#-------------------------------------- FEATURE ENGINEERING --------------------------

VECTORIZERS = {
    'BoW': CountVectorizer(),
    'TF-IDF': TfidfVectorizer()
}

ALGORITHMS = {
    'LogisticRegression': LogisticRegression(),
    'MultinomialNB': MultinomialNB(),
    'XGBoost': XGBClassifier(),
    'RandomForest': RandomForestClassifier(),
    'GradientBoosting': GradientBoostingClassifier()
}


# ------------------ TRAINING AND EVALUATE MODELS------------------------

def train_and_evaluate(df):    
    with mlflow.start_run(run_name='All Experiments') as parent_run:
        for algo_name, algorithm in ALGORITHMS.items():
            for vec_name, vectorizer in VECTORIZERS.items():
                with mlflow.start_run(run_name=f"{algo_name} with {vec_name}", nested=True) as child_run:
                    try:
                        # FEATURE EXTRACTION
                        X = vectorizer.fit_transform(df['review'])            
                        y = df['sentiment']
                        
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=CONFIG["test_size"], random_state=42)
                        
                        # LOG PREPROCESSING PARAMETERS
                        mlflow.log_params({
                            "vectorizer": vec_name,
                            "algorithm": algo_name,
                            "test_size": CONFIG["test_size"]
                        })
                        
                        # TRAIN MODEL
                        model = algorithm
                        model.fit(X_train, y_train)
                        
                        # LOG MODEL PARAMETERS
                        log_model_params(algo_name, model)
                        
                        # EVALUATE MODEL
                        y_pred = model.predict(X_test)
                        metrics = {
                            'accuracy': accuracy_score(y_test, y_pred),
                            'precision': precision_score(y_test, y_pred),
                            "recall": recall_score(y_test, y_pred),
                            "f1_score": f1_score(y_test, y_pred)
                        }
                        
                        mlflow.log_metrics(metrics)
                        
                        # LOG MODEL
                        input_example = X_test[:5].toarray() if scipy.sparse.issparse(X_test) else X_test[:5]
                        mlflow.sklearn.log_model(model, "model", input_example=input_example)
                        
                        # PRINT RESULTS FOR VERIFICATION
                        print(f"\nAlgorithm: {algo_name}, Vectorizer: {vec_name}")
                        print(f"Metrics: {metrics}")
                        
                    except Exception as e:
                        print(f"Error in training: {algo_name} with {vec_name}: {e}")
                        mlflow.log_text(str(e), "error_log.txt")


def log_model_params(algo_name, model):
    """Logs hyperparameters of the trained model to MLflow."""
    params_to_log = {}
    
    if algo_name == "LogisticRegression":
        params_to_log["C"] = model.C 
        
    elif algo_name == "MultinomialNB":
        params_to_log["alpha"] = model.alpha
        
    elif algo_name == "XGBoost":
        params_to_log["n_estimators"] = model.n_estimators
        params_to_log["learning_rate"] = model.learning_rate
        
    elif algo_name == "RandomForest":
        params_to_log["n_estimators"] = model.n_estimators
        params_to_log["max_depth"] = model.max_depth
        
    elif algo_name == "GradientBoosting":
        params_to_log["n_estimators"] = model.n_estimators
        params_to_log['learning_rate'] = model.learning_rate
        params_to_log['max_depth'] = model.max_depth
        
    mlflow.log_params(params_to_log)


# ---------------- EXECUTION ------------------------- 

if __name__ == "__main__":
    df = load_data(CONFIG["data_path"])
    train_and_evaluate(df)