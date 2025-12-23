import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
import mlflow
import mlflow.sklearn
import dagshub

# --- 1. SETUP AUTHENTICATION ---
repo_owner = 'AzizSuryaPradana'
repo_name = 'submission_exam_AzizSuryaPradana'
token = os.getenv("DAGSHUB_CLIENT_TOKEN")

if token:
    os.environ["MLFLOW_TRACKING_USERNAME"] = token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    dagshub.auth.add_app_token(token)

# --- 2. INITIALIZE TRACKING ---
# Kita pindahkan inisialisasi agar MLflow benar-benar sinkron dengan DagsHub
mlflow.set_tracking_uri(f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow")
dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
mlflow.set_experiment("Exam_Score_Regression_Experiment")

# --- 3. LOAD DATASET ---
csv_path = 'Exam_Score_Preprocessed.csv'
if not os.path.exists(csv_path):
    csv_path = os.path.join('MLProject', 'Exam_Score_Preprocessed.csv')

df = pd.read_csv(csv_path)
print(f"✅ Dataset berhasil dimuat: {csv_path}")

# --- 4. TRAINING ---
X = df.drop(columns=['exam_score'])
y = df['exam_score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SOLUSI FINAL: Matikan pendeteksian run ID otomatis dari environment 
# untuk mencegah MLflow mencari ID lokal di server remote.
if "MLFLOW_RUN_ID" in os.environ:
    del os.environ["MLFLOW_RUN_ID"]

with mlflow.start_run(run_name="RandomForest_Retraining_Official") as run:
    # Aktifkan autologging
    mlflow.sklearn.autolog()
    
    print(f"🚀 Memulai Retraining... (Run ID: {run.info.run_id})")
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Log Manual Metrik & Artefak (Syarat Advance)
    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    mlflow.log_metric("rmse_final", rmse)
    
    # Save & Log artifacts
    summary_path = "dataset_summary.csv"
    df.describe().to_csv(summary_path)
    mlflow.log_artifact(summary_path)
    
    print(f"✅ Retraining Berhasil! RMSE: {rmse:.4f}")