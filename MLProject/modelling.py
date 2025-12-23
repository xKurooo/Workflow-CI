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
# Menggunakan Secret dari GitHub Actions jika ada, jika tidak pakai string hardcoded (untuk lokal)
TOKEN_ASLI = os.getenv("DAGSHUB_CLIENT_TOKEN") or "33b1311e98312d1cf9d695883fa1bfc72556d5a3"
os.environ["MLFLOW_TRACKING_USERNAME"] = "AzizSuryaPradana"
os.environ["MLFLOW_TRACKING_PASSWORD"] = TOKEN_ASLI

repo_owner = 'AzizSuryaPradana' 
repo_name = 'submission_exam_AzizSuryaPradana'

try:
    dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
    mlflow.set_tracking_uri(f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow")
except Exception as e:
    print(f"DagsHub init dilewati/error: {e}")

# --- 3. LOAD DATASET ---
# Path fleksibel untuk lokal maupun GitHub Actions
csv_path = 'Exam_Score_Preprocessed.csv'
if not os.path.exists(csv_path):
    csv_path = os.path.join('MLProject', 'Exam_Score_Preprocessed.csv')

try:
    df_model = pd.read_csv(csv_path)
    print(f"✅ Dataset berhasil dimuat dari {csv_path}")
except Exception as e:
    print(f"❌ Error: File dataset tidak ditemukan! {e}")
    sys.exit(1)

# --- 4. TRAINING & LOGGING ---
X = df_model.drop(columns=['exam_score'])
y = df_model['exam_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("Exam_Score_Regression_Experiment")

# PENTING: Mengambil Run ID dari environment variable yang disediakan MLflow CLI
# Ini mencegah error "Run with id=... not found"
active_run_id = os.getenv("MLFLOW_RUN_ID")

with mlflow.start_run(run_id=active_run_id, run_name="RandomForest_Regressor_Base") as run:
    # Aktifkan autologging
    mlflow.sklearn.autolog()
    
    print("Mulai training model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    # Evaluasi
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred) 
    
    print(f"\n--- HASIL EVALUASI ---")
    print(f"📊 R2 Score: {r2:.4f}")
    print(f"📉 MAE: {mae:.4f}")
    print(f"📉 RMSE: {rmse:.4f}")
    
    # Log metrik tambahan secara manual (jika diperlukan)
    mlflow.log_metric("custom_rmse", rmse)
    
    print(f"\n✅ Training Selesai! Run ID: {run.info.run_id}")