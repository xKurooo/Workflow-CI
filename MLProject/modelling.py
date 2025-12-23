import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
import mlflow
import mlflow.sklearn
import dagshub

# --- 1. SETUP AUTHENTICATION (ANTI-OAUTH UNTUK CI) ---
repo_owner = 'AzizSuryaPradana'
repo_name = 'submission_exam_AzizSuryaPradana'

# Mengambil token dari environment variable GitHub Secrets
token = os.getenv("DAGSHUB_CLIENT_TOKEN")

if token:
    # Set kredensial secara manual agar tidak membuka browser login
    os.environ["MLFLOW_TRACKING_USERNAME"] = token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    try:
        dagshub.auth.add_app_token(token)
    except Exception as e:
        print(f"Token setup warning: {e}")

# Inisialisasi DagsHub
dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
mlflow.set_tracking_uri(f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow")
mlflow.set_experiment("Exam_Score_Regression_Experiment")

# --- 2. LOAD DATASET ---
# Mencari file di folder saat ini
csv_path = 'Exam_Score_Preprocessed.csv'
if not os.path.exists(csv_path):
    csv_path = os.path.join('MLProject', 'Exam_Score_Preprocessed.csv')

try:
    df = pd.read_csv(csv_path)
    print(f"✅ Dataset berhasil dimuat dari {csv_path}")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    sys.exit(1)

# --- 3. PREPARATION & TRAINING ---
X = df.drop(columns=['exam_score'])
y = df['exam_score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logika mendeteksi run aktif dari MLflow CLI
active_run = mlflow.active_run()
run_context = mlflow.start_run(run_id=active_run.info.run_id) if active_run else mlflow.start_run(run_name="RandomForest_Retraining_CI")

with run_context as run:
    # Aktifkan autologging
    mlflow.sklearn.autolog()
    
    print(f"🚀 Memulai Retraining... Run ID: {run.info.run_id}")
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluasi
    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Log metrik tambahan secara manual
    mlflow.log_metric("rmse_score", rmse)
    
    # --- 4. ARTIFACT LOGGING (Min. 2 untuk Advance) ---
    # Artefak 1: Ringkasan Dataset
    df.describe().to_csv("dataset_summary.csv")
    mlflow.log_artifact("dataset_summary.csv")
    
    # Artefak 2: Feature Importance
    feat_importances = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
    feat_importances.to_csv("feature_importance.csv", index=False)
    mlflow.log_artifact("feature_importance.csv")

    print(f"✅ Retraining selesai! R2 Score: {r2:.4f}")