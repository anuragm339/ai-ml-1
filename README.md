# Tourism Wellness Package — MLOps Pipeline

**Visit with Us** | Predicting customer purchase behaviour for the Wellness Tourism Package

## Live Links
| Resource | URL |
|---|---|
| Streamlit App (HF Space) | https://huggingface.co/spaces/anuragmishrarock/tourism-wellness-app |
| Dataset on HF | https://huggingface.co/datasets/anuragmishrarock/tourism-wellness-dataset |
| Model on HF | https://huggingface.co/anuragmishrarock/tourism-wellness-model |
| GitHub Repo | https://github.com/anuragm339/ai-ml-1 |

## Project Structure
```
├── tourism_wellness_mlops.ipynb   # Main notebook (all sections executed)
├── tourism.csv                    # Raw dataset
├── Visit_with_Us/
│   ├── data/                      # Cleaned, train & test CSVs
│   ├── models/                    # Best model (Gradient Boosting) + feature names
│   ├── plots/                     # EDA & evaluation plots
│   ├── deployment/
│   │   ├── app.py                 # Streamlit app
│   │   ├── Dockerfile             # Docker config for HF Spaces
│   │   ├── requirements.txt       # Python dependencies
│   │   └── deploy_to_hf.py        # Hosting script
│   └── .github/workflows/
│       └── pipeline.yml           # GitHub Actions CI/CD pipeline
```

## Best Model Results
| Metric | Score |
|---|---|
| Algorithm | Gradient Boosting |
| ROC-AUC | 0.9534 |
| Accuracy | 93.3% |
| F1-Score | 0.811 |
| Precision | 0.894 |
| Recall | 0.742 |

## Pipeline Steps
1. **Data Registration** — Upload raw data to Hugging Face dataset repo
2. **Data Preparation** — Clean, encode, split 80/20, upload train/test to HF
3. **Model Building** — Train 5 models (DT, RF, AdaBoost, GBM, XGBoost) with GridSearchCV + MLflow tracking
4. **Model Deployment** — Register best model on HF Model Hub, deploy Streamlit app to HF Spaces
5. **CI/CD** — GitHub Actions automates the full pipeline on every push to `main`

## GitHub Actions
The `.github/workflows/pipeline.yml` automates:
- Data preparation
- Model training & evaluation
- Deployment to Hugging Face Spaces

Add `HF_TOKEN` to GitHub repo Secrets to activate the workflow.
