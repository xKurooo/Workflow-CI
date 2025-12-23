# --- modelling.py ---

# 1. WORKAROUND UNTUK PYTHON 3.12+ (Fix Error distutils)
import sys
import types

try:
    import distutils.version
except ImportError:
    d = types.ModuleType("distutils")
    sys.modules["distutils"] = d
    sys.modules["distutils.version"] = types.ModuleType("distutils.version")

# 2. IMPORT LIBRARY UTAMA
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
# Tambahkan root_mean_squared_error di sini
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, root_mean_squared_error
import mlflow
import mlflow.sklearn
import dagshub
import os

# --- 1. SETUP AUTHENTICATION ---
os.environ["MLFLOW_TRACKING_USERNAME"] = "AzizSuryaPradana"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "33b1311e98312d1cf9d695883fa1bfc72556d5a3" 

repo_owner = 'AzizSuryaPradana' 
repo_name = 'submission_exam_AzizSuryaPradana'

try:
    dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
    mlflow.set_tracking_uri(f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow")
except Exception as e:
    print(f"DagsHub init dilewati: {e}")

# --- 2. LOAD PREPROCESSED DATASET ---
try:
    df_model = pd.read_csv('Exam_Score_Preprocessed.csv')
    print("✅ Dataset berhasil dimuat.")
except FileNotFoundError:
    print("❌ Error: File 'Exam_Score_Preprocessed.csv' tidak ditemukan!")
    sys.exit()

# --- 3. TRAINING & LOGGING ---
X = df_model.drop(columns=['exam_score'])
y = df_model['exam_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("Exam_Score_Regression_Experiment")

with mlflow.start_run(run_name="RandomForest_Regressor_Base"):
    mlflow.sklearn.autolog()
    
    print("Mulai training model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    # --- PERBAIKAN EVALUASI ---
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    # Gunakan fungsi baru untuk RMSE
    rmse = root_mean_squared_error(y_test, y_pred) 
    
    print(f"\n--- HASIL EVALUASI ---")
    print(f"📊 R2 Score: {r2:.4f}")
    print(f"📉 MAE: {mae:.4f}")
    print(f"📉 RMSE: {rmse:.4f}")
    
    mlflow.log_metric("rmse", rmse)
    
    print(f"\n✅ Training Selesai! Cek dashboard di DagsHub.")