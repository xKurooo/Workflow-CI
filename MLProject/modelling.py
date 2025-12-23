# --- modelling.py ---

import sys
import types
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
import mlflow
import mlflow.sklearn
import dagshub

# 1. WORKAROUND UNTUK PYTHON 3.12+ (Fix Error distutils)
try:
    import distutils.version
except ImportError:
    d = types.ModuleType("distutils")
    sys.modules["distutils"] = d
    sys.modules["distutils.version"] = types.ModuleType("distutils.version")

# --- 2. SETUP AUTHENTICATION ---
TOKEN_ASLI = os.getenv("DAGSHUB_CLIENT_TOKEN") or "33b1311e98312d1cf9d695883fa1bfc72556d5a3"
os.environ["MLFLOW_TRACKING_USERNAME"] = "AzizSuryaPradana"
os.environ["MLFLOW_TRACKING_PASSWORD"] = TOKEN_ASLI

repo_owner = 'AzizSuryaPradana' 
repo_name = 'submission_exam_AzizSuryaPradana'

try:
    dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
    mlflow.set_tracking_uri(f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow")
    # Set experiment agar sinkron antara lokal dan remote
    mlflow.set_experiment("Exam_Score_Regression_Experiment")
except Exception as e:
    print(f"DagsHub init warning: {e}")

# --- 3. LOAD DATASET ---
csv_path = 'Exam_Score_Preprocessed.csv'
if not os.path.exists(csv_path):
    csv_path = os.path.join('MLProject', 'Exam_Score_Preprocessed.csv')

try:
    df_model = pd.read_csv(csv_path)
    print(f"✅ Dataset berhasil dimuat: {csv_path}")
except Exception as e:
    print(f"❌ Error: Dataset tidak ditemukan! {e}")
    sys.exit(1)

# --- 4. TRAINING & LOGGING ---
X = df_model.drop(columns=['exam_score'])
y = df_model['exam_score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MODIFIKASI KRUSIAL: 
# Kita tidak memasukkan run_id secara eksplisit. 
# MLflow akan otomatis mendeteksi jika dijalankan di dalam 'mlflow run'.
with mlflow.start_run(run_name="RandomForest_Regressor_Base") as run:
    # Aktifkan autologging untuk menangkap semua parameter otomatis
    mlflow.sklearn.autolog()
    
    print(f"🚀 Training dimulai... Run ID: {run.info.run_id}")
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    # Metrik manual agar tampil jelas di dashboard
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred) 
    
    mlflow.log_metrics({"r2_score": r2, "mae": mae, "rmse": rmse})
    
    print(f"📊 Evaluasi - R2: {r2:.4f}, MAE: {mae:.4f}")
    print(f"✅ Berhasil log ke DagsHub!")