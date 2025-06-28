import os
import mlflow
import mlflow.sklearn
import boto3
from dotenv import load_dotenv
import pickle
import numpy as np
from sklearn.svm import SVC # NEW: Import SVC
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

# SVM specific hyperparameters
C_val = float(os.environ.get("C", 1.0))
kernel = os.environ.get("KERNEL", "rbf")
gamma_val_str = os.environ.get("GAMMA", "scale") # 'scale', 'auto', or float
gamma_val = gamma_val_str
if gamma_val_str.replace('.', '', 1).isdigit(): # Check if it's a number
    gamma_val = float(gamma_val_str)
degree = int(os.environ.get("DEGREE", 3)) # Only for 'poly' kernel

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
    mlflow.log_param("model_type", "SVC") # Log the specific model type
    mlflow.log_param("random_state", common_random_state)
    mlflow.log_param("cv", common_cv)

    # Log SVM specific hyperparameters
    mlflow.log_param("C", C_val)
    mlflow.log_param("kernel", kernel)
    mlflow.log_param("gamma", gamma_val)
    mlflow.log_param("degree", degree)

    print(f"Training SVC with C={C_val}, kernel='{kernel}', gamma='{gamma_val}', degree={degree}...")

    # Initialize StratifiedKFold for cross-validation
    skf = StratifiedKFold(n_splits=common_cv, shuffle=True, random_state=common_random_state)

    accuracy_scores = []
    f1_scores = []

    # Perform cross-validation
    for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Instantiate SVM model
        model = SVC(C=C_val, kernel=kernel, gamma=gamma_val, degree=degree, random_state=common_random_state)
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
    final_model = SVC(C=C_val, kernel=kernel, gamma=gamma_val, degree=degree, random_state=common_random_state)
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
