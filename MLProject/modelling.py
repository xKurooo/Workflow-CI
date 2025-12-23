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

# Fix untuk Python 3.12+
try:
    import distutils.version
except ImportError:
    d = types.ModuleType("distutils")
    sys.modules["distutils"] = d
    sys.modules["distutils.version"] = types.ModuleType("distutils.version")

# --- AUTHENTICATION ---
TOKEN = os.getenv("DAGSHUB_CLIENT_TOKEN") or "33b1311e98312d1cf9d695883fa1bfc72556d5a3"
os.environ["MLFLOW_TRACKING_USERNAME"] = "AzizSuryaPradana"
os.environ["MLFLOW_TRACKING_PASSWORD"] = TOKEN

repo_owner = 'AzizSuryaPradana'
repo_name = 'submission_exam_AzizSuryaPradana'

try:
    dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
    mlflow.set_tracking_uri(f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow")
    mlflow.set_experiment("Exam_Score_Regression_Experiment")
except Exception as e:
    print(f"Warning DagsHub: {e}")

# --- LOAD DATA ---
# Mencari file di folder saat ini (penting untuk CI)
csv_path = 'Exam_Score_Preprocessed.csv'
if not os.path.exists(csv_path):
    csv_path = os.path.join('MLProject', 'Exam_Score_Preprocessed.csv')

df = pd.read_csv(csv_path)
print(f"✅ Dataset loaded from: {csv_path}")

# --- MODELLING ---
X = df.drop(columns=['exam_score'])
y = df['exam_score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Menggunakan nested=True untuk menghindari RESOURCE_DOES_NOT_EXIST
with mlflow.start_run(run_name="RandomForest_CI_Run", nested=True) as run:
    mlflow.sklearn.autolog()
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = root_mean_squared_error(y_test, predictions)
    
    mlflow.log_metrics({"r2_score": r2, "mae": mae, "rmse": rmse})
    
    print(f"🚀 Training Success! Run ID: {run.info.run_id}")
    print(f"📊 R2 Score: {r2:.4f}")