# AI Recruitment Guardian

An Explainable Machine Learning Framework for Fake Job Detection and Risk Assessment.

## Team Members

- Ananaya Arora
- Vanshika

## Project Objectives

- Detect fraudulent job postings using Machine Learning.
- Compare multiple ML algorithms.
- Generate a Job Trust Score.
- Explain predictions using SHAP.
- Build an interactive web application for real-time verification.

## Project Structure

```
data/
    raw/                    fake_job_postings.csv (EMSCAD dataset)

docs/                       problem statement, literature review

notebooks/
    01_Data_Understanding.ipynb
    02_Data_Preprocessing.ipynb
    03_Model_Development.ipynb
    04_Explainable_AI.ipynb

reports/                    evaluation plots, SHAP plots, comparison table

templates/                  Flask HTML template
static/                     Flask CSS/JS assets

server.py                   Flask backend (web app)
ml_pipeline.py              shared prediction + explainability logic
requirements.txt            Python dependencies

best_model.pkl              tuned XGBoost model (best of 5 compared)
all_tuned_models.pkl        all 5 tuned models, for comparison
tfidf_vectorizer.pkl        fitted TF-IDF vectorizer
onehot_encoder.pkl          fitted OneHotEncoder
train_test_split.pkl        cached train/test split
```

## Current Status

- Literature Review ✅
- Project Proposal ✅
- Data Understanding ✅
- Preprocessing ✅
- Model Development (5 models trained, tuned, and compared) ✅
- Explainable AI (SHAP, Job Trust Score, suspicious phrase highlighting) ✅
- Web Application (Flask) ✅

## Results

Five models were trained and hyperparameter-tuned; XGBoost was selected as the
best performer based on F1-score and ROC-AUC on the held-out test set (see
`reports/model_comparison_table.csv`).

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|---|
| **XGBoost** | 0.987 | 0.920 | 0.798 | **0.854** | 0.993 |
| SVM | 0.986 | 0.949 | 0.751 | 0.839 | 0.985 |
| Random Forest | 0.983 | 0.925 | 0.711 | 0.804 | 0.994 |
| Logistic Regression | 0.978 | 0.720 | 0.879 | 0.792 | 0.982 |
| Naive Bayes | 0.967 | 0.665 | 0.630 | 0.647 | 0.955 |

## Running the Web Application

```bash
pip install -r requirements.txt
python server.py
```

Open `http://localhost:5000` in your browser. The app accepts a job posting
either pasted directly or fetched from a URL, and returns a Job Trust Score
(0–100) with SHAP-based reasoning for the prediction.

## Tech Stack

Python · scikit-learn · XGBoost · SHAP · Flask · pandas · TF-IDF
