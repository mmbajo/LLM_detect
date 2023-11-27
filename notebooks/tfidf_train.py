import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold


from joblib import dump
from pathlib import Path


def train(train_data, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    
    def cleaning(dataset):
        dataset['text'] = dataset['text'].str.strip()
        dataset["text"] = dataset["text"].replace('\\n',' ')
        # dataset["text"] = dataset["text"].str.split('ubject: ').str[-1].str.strip()
        # dataset["text"] = dataset["text"].str.split('Zip').str[-1].str.strip()
        # dataset["text"] = dataset["text"].str.split('ZIP').str[-1].str.strip()

        # dataset = dataset.rename(columns = {'generated':'label'})

        return dataset
    
    train = pd.read_csv(train_data, sep=",")
    train = cleaning(train)
    
    skf = StratifiedKFold(n_splits=16, shuffle=True, random_state=42)
    X = train.loc[:, train.columns != "label"]
    y = train.loc[:, train.columns == "label"]

    for i, (train_index, valid_index) in enumerate(skf.split(X, y)):
        train.loc[valid_index, "fold"] = i
        
    print(train.groupby("fold")["label"].value_counts())
        
    test = train[train["fold"] == 15]
    train = train[train["fold"] != 15]
    
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), 
                                 sublinear_tf=True)
    X = vectorizer.fit_transform(train["text"])
    
    dump(vectorizer, out_dir / 'tfidf_vectorizer.joblib')
    
    svc_model = SVC(probability=True)
    lr_model = LogisticRegression(solver="liblinear")
    sgd_model = SGDClassifier(max_iter=1000, tol=1e-3, loss="modified_huber")
    xgb_model = XGBClassifier(n_estimators=550, max_depth=5, random_state=42, n_jobs=-1)
    lgb_model = LGBMClassifier(n_estimators=550, max_depth=5, random_state=42, n_jobs=-1)
    
    models = [
        # ('svc', svc_model),
        ('lr', lr_model),
        ('sgd', sgd_model),
        # ('xgb', xgb_model),
        # ('lgb', lgb_model)
    ]
    
    weights = [0.01, 0.99]
    
    ensemble = VotingClassifier(estimators=models, voting='soft', weights=weights, verbose=True, n_jobs=-1)
    ensemble.fit(X, train.label)
    
    dump(ensemble, out_dir / 'ensemble.joblib')
    X_test = vectorizer.transform(test["text"])
    preds_test = ensemble.predict_proba(X_test)[:,1]
    preds_test_hard = preds_test >= 0.5
    auc = roc_auc_score(test["label"].values, preds_test)
    acc = accuracy_score(test["label"].values, preds_test_hard)
    print("AUC: ", auc)
    print("ACC: ", acc)
    
    
if __name__ == "__main__":
    import argparse
    
    # Create the parser
    parser = argparse.ArgumentParser(description='Script to run training.')

    # Add arguments
    parser.add_argument('--data_path', type=str, required=True, help='Path to the training data')
    parser.add_argument('--out_dir', type=str, required=True, help='Path to the dumping directory')

    # Parse the arguments
    args = parser.parse_args()

    # Extract values from args and run the infer function
    DATA_PATH = args.data_path
    OUT_DIR = args.out_dir
    
    train(DATA_PATH, OUT_DIR)
    