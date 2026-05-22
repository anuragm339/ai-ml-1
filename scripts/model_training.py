import os
import json
import pickle
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from huggingface_hub import HfApi, login
from datasets import load_dataset

HF_TOKEN     = os.environ.get("HF_TOKEN")
HF_USERNAME  = "anuragmishrarock"
DATASET_REPO = f"{HF_USERNAME}/tourism-wellness-dataset"
MODEL_REPO   = f"{HF_USERNAME}/tourism-wellness-model"

login(token=HF_TOKEN, add_to_git_credential=False)
api = HfApi()

os.makedirs("Visit_with_Us/models", exist_ok=True)

# Load from HF
train_ds = load_dataset(DATASET_REPO, data_files="train.csv", split="train")
test_ds  = load_dataset(DATASET_REPO, data_files="test.csv",  split="train")
X_train  = train_ds.to_pandas().drop(columns=["ProdTaken"])
y_train  = train_ds.to_pandas()["ProdTaken"]
X_test   = test_ds.to_pandas().drop(columns=["ProdTaken"])
y_test   = test_ds.to_pandas()["ProdTaken"]
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

models = {
    "Decision Tree":     (DecisionTreeClassifier(random_state=42),
                          {"max_depth":[3,5,7],"min_samples_split":[2,5],"criterion":["gini","entropy"]}),
    "Random Forest":     (RandomForestClassifier(random_state=42),
                          {"n_estimators":[50,100],"max_depth":[5,7]}),
    "AdaBoost":          (AdaBoostClassifier(random_state=42),
                          {"n_estimators":[50,100],"learning_rate":[0.1,1.0]}),
    "Gradient Boosting": (GradientBoostingClassifier(random_state=42),
                          {"n_estimators":[50,100],"learning_rate":[0.1,0.2],"max_depth":[3,5]}),
    "XGBoost":           (XGBClassifier(random_state=42, eval_metric="logloss", verbosity=0),
                          {"n_estimators":[50,100],"learning_rate":[0.1,0.2],"max_depth":[3,5]}),
}

mlflow.set_experiment("Tourism_Wellness_CI")
best_model, best_score, best_name = None, 0, ""

for name, (model, params) in models.items():
    print(f"Training {name}...")
    with mlflow.start_run(run_name=name):
        gs = GridSearchCV(model, params, cv=5, scoring="roc_auc", n_jobs=-1)
        gs.fit(X_train, y_train)
        est   = gs.best_estimator_
        y_pred = est.predict(X_test)
        y_prob = est.predict_proba(X_test)[:, 1]
        auc    = roc_auc_score(y_test, y_prob)
        mlflow.log_params(gs.best_params_)
        mlflow.log_metrics({"accuracy": accuracy_score(y_test, y_pred), "roc_auc": auc,
                            "f1": f1_score(y_test, y_pred),
                            "precision": precision_score(y_test, y_pred),
                            "recall": recall_score(y_test, y_pred)})
        mlflow.sklearn.log_model(est, "model")
        print(f"  ROC-AUC: {auc:.4f}")
        if auc > best_score:
            best_score, best_model, best_name = auc, est, name

print(f"\nBest: {best_name} (AUC={best_score:.4f})")

# Save and upload best model
model_path = "Visit_with_Us/models/best_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(best_model, f)

with open("Visit_with_Us/models/feature_names.json", "w") as f:
    json.dump(list(X_train.columns), f)

api.create_repo(repo_id=MODEL_REPO, repo_type="model", private=False, exist_ok=True)
for fname in ["best_model.pkl", "feature_names.json"]:
    api.upload_file(path_or_fileobj=f"Visit_with_Us/models/{fname}",
                    path_in_repo=fname, repo_id=MODEL_REPO, repo_type="model")
    print(f"Uploaded {fname} to HF Model Hub")
