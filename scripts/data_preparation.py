import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import HfApi, login
from datasets import load_dataset

HF_TOKEN     = os.environ.get("HF_TOKEN")
HF_USERNAME  = "anuragmishrarock"
DATASET_REPO = f"{HF_USERNAME}/tourism-wellness-dataset"

login(token=HF_TOKEN, add_to_git_credential=False)
api = HfApi()

os.makedirs("Visit_with_Us/data", exist_ok=True)

# Load raw data from HF
ds = load_dataset(DATASET_REPO, data_files="tourism.csv", split="train")
df = ds.to_pandas()
print(f"Loaded: {df.shape}")

# Clean
df.drop(columns=[c for c in df.columns if "Unnamed" in c], inplace=True)
df.drop(columns=["CustomerID"], inplace=True, errors="ignore")
df["Gender"] = df["Gender"].replace("Fe Male", "Female")

for col in df.select_dtypes(include=[np.number]).columns:
    df[col].fillna(df[col].median(), inplace=True)
for col in df.select_dtypes(include="object").columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Encode
le = LabelEncoder()
for col in ["TypeofContact", "Gender"]:
    df[col] = le.fit_transform(df[col].astype(str))
df = pd.get_dummies(df, columns=["Occupation", "ProductPitched", "MaritalStatus", "Designation"])

# Split
X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

train_df = pd.concat([X_train, y_train], axis=1)
test_df  = pd.concat([X_test,  y_test],  axis=1)
train_df.to_csv("Visit_with_Us/data/train.csv", index=False)
test_df.to_csv("Visit_with_Us/data/test.csv",   index=False)
print(f"Train: {train_df.shape}, Test: {test_df.shape}")

# Upload to HF
for fname in ["train.csv", "test.csv"]:
    api.upload_file(path_or_fileobj=f"Visit_with_Us/data/{fname}",
                    path_in_repo=fname, repo_id=DATASET_REPO, repo_type="dataset")
    print(f"Uploaded {fname} to HF")
