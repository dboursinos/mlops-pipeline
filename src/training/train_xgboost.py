import os
import mlflow
import mlflow.sklearn
import boto3
from dotenv import load_dotenv
import pickle
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

# --- MinIO Setup ---
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY")
MINIO_BUCKET = os.environ.get("MINIO_BUCKET")

if not all([MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_BUCKET]):
    print("Error: Not all MinIO credentials are provided as environment variables.")
    exit(1)

s3_client = boto3.client(
    "s3",
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY,
)

# --- Hyperparameter Loading ---
common_random_state = int(os.environ.get("RANDOM_STATE", 42))
common_cv = int(os.environ.get("CV", 5))

# XGBoost specific hyperparameters
n_estimators = int(os.environ.get("N_ESTIMATORS", 100))
learning_rate = float(os.environ.get("LEARNING_RATE", 0.1))
max_depth_str = os.environ.get("MAX_DEPTH", "6")
max_depth = int(max_depth_str) if max_depth_str.lower() != "none" else None
subsample = float(os.environ.get("SUBSAMPLE", 1.0))
colsample_bytree = float(os.environ.get("COLSAMPLE_BYTREE", 1.0))
gamma = float(os.environ.get("GAMMA", 0))
reg_alpha = float(os.environ.get("REG_ALPHA", 0))
reg_lambda = float(os.environ.get("REG_LAMBDA", 1))

# --- Data Loading ---
X_TRAIN_KEY = os.environ.get("X_TRAIN_KEY", None)
Y_TRAIN_KEY = os.environ.get("Y_TRAIN_KEY", None)
X_TEST_KEY = os.environ.get("X_TEST_KEY", None)
Y_TEST_KEY = os.environ.get("Y_TEST_KEY", None)

if not all([X_TRAIN_KEY, Y_TRAIN_KEY, X_TEST_KEY, Y_TEST_KEY]):
    print("Error: Not all data file keys are provided as environment variables.")
    exit(1)

def load_data_from_minio(key):
    """Loads data from MinIO."""
    try:
        response = s3_client.get_object(Bucket=MINIO_BUCKET, Key=key)
        data = pickle.loads(response["Body"].read())
        print(f"Loaded {key} from MinIO")
        return data
    except Exception as e:
        print(f"Error loading {key} from MinIO: {e}")
        return None

X = load_data_from_minio(X_TRAIN_KEY)
y = load_data_from_minio(Y_TRAIN_KEY)
X_test = load_data_from_minio(X_TEST_KEY)
y_test = load_data_from_minio(Y_TEST_KEY)

if X is None or y is None or X_test is None or y_test is None:
    print("Failed to load all necessary data. Exiting.")
    exit(1)

# --- MLflow Setup ---
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME"))

# --- Main Training Logic ---
with mlflow.start_run():
    mlflow.log_param("model_type", "XGBoostClassifier") # Log the specific model type
    mlflow.log_param("random_state", common_random_state)
    mlflow.log_param("cv", common_cv)

    # Log XGBoost specific hyperparameters
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("subsample", subsample)
    mlflow.log_param("colsample_bytree", colsample_bytree)
    mlflow.log_param("gamma", gamma)
    mlflow.log_param("reg_alpha", reg_alpha)
    mlflow.log_param("reg_lambda", reg_lambda)

    print(f"Training XGBoost Classifier with n_estimators={n_estimators}, learning_rate={learning_rate}, max_depth={max_depth}...")

    # Initialize StratifiedKFold for cross-validation
    skf = StratifiedKFold(n_splits=common_cv, shuffle=True, random_state=common_random_state)

    accuracy_scores = []
    f1_scores = []

    # Perform cross-validation
    for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Instantiate XGBoost model
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            gamma=gamma,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=common_random_state,
            use_label_encoder=False, # Suppress XGBoost deprecation warning
            eval_metric='logloss'    # Suppress XGBoost deprecation warning
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)

        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average="weighted")
        accuracy_scores.append(accuracy)
        f1_scores.append(f1)

        mlflow.log_metric(f"accuracy_fold_{fold}", accuracy)
        mlflow.log_metric(f"f1_fold_{fold}", f1)

    avg_accuracy = np.mean(accuracy_scores)
    avg_f1 = np.mean(f1_scores)
    mlflow.log_metric("accuracy_avg", avg_accuracy)
    mlflow.log_metric("f1_avg", avg_f1)

    # Train final model on the entire training set
    final_model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        gamma=gamma,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=common_random_state,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    final_model.fit(X, y)

    y_pred_train = final_model.predict(X)
    train_accuracy = accuracy_score(y, y_pred_train)
    train_f1 = f1_score(y, y_pred_train, average="weighted")
    mlflow.log_metric("train_accuracy", train_accuracy)
    mlflow.log_metric("train_f1", train_f1)

    y_pred_test = final_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test, average="weighted")
    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_metric("test_f1", test_f1)

    mlflow.sklearn.log_model(final_model, "model")

    print(f"model saved to: {mlflow.get_artifact_uri('model')}")
