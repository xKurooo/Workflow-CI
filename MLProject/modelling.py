import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
import mlflow
import mlflow.sklearn
import dagshub

# --- SETUP DAGSHUB & MLFLOW ---
repo_owner = 'AzizSuryaPradana'
repo_name = 'submission_exam_AzizSuryaPradana'
dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
mlflow.set_tracking_uri(f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow")
mlflow.set_experiment("Exam_Score_Regression_Experiment")

# --- LOAD DATA ---
df = pd.read_csv('Exam_Score_Preprocessed.csv')

# --- TRAINING ---
X = df.drop(columns=['exam_score'])
y = df['exam_score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# LOGIKA KRUSIAL: Gunakan run yang sudah ada jika dijalankan via 'mlflow run'
active_run = mlflow.active_run()
run_context = mlflow.start_run(run_id=active_run.info.run_id) if active_run else mlflow.start_run(run_name="RandomForest_CI")

with run_context as run:
    mlflow.sklearn.autolog()
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Log Artifacts (Min. 2 untuk Advance)
    df.describe().to_csv("summary.csv")
    mlflow.log_artifact("summary.csv")
    mlflow.log_metric("rmse", root_mean_squared_error(y_test, model.predict(X_test)))
    
    print(f"✅ Retraining Berhasil via MLflow Run. ID: {run.info.run_id}")