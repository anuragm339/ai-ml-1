---
language: en
tags: [tabular-classification, tourism, wellness, sklearn, xgboost]
datasets: [anuragmishrarock/tourism-wellness-dataset]
metrics: [roc_auc, accuracy, f1]
---
# Tourism Wellness Package Purchase Prediction

**Best Model**: Gradient Boosting  
**ROC-AUC**: 0.9534  
**Accuracy**: 0.9334  
**F1-Score**: 0.8110  

## Training Data
- Dataset: [anuragmishrarock/tourism-wellness-dataset](https://huggingface.co/datasets/anuragmishrarock/tourism-wellness-dataset)
- Train samples: 3302 | Test samples: 826

## Models Evaluated
Decision Tree, Random Forest, AdaBoost, Gradient Boosting, XGBoost

## Usage
```python
import pickle, json
from huggingface_hub import hf_hub_download
model = pickle.load(open(hf_hub_download('anuragmishrarock/tourism-wellness-model', 'model.pkl'), 'rb'))
features = json.load(open(hf_hub_download('anuragmishrarock/tourism-wellness-model', 'feature_names.json')))
```
