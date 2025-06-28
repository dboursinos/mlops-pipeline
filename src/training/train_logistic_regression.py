import os
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
import boto3
from dotenv import load_dotenv
import pickle
import numpy as np

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

# Define hyperparameters with default values
C = float(os.environ.get("C", 1.0))  # Regularization strength
random_state = int(
    os.environ.get("RANDOM_STATE", 42)
)  # Random state for reproducibility
cv = int(os.environ.get("CV", 5))  # Number of cross-validation folds

# Get data file keys from environment variables
X_TRAIN = os.environ.get("X_TRAIN_KEY", None)
Y_TRAIN = os.environ.get("Y_TRAIN_KEY", None)
X_TEST = os.environ.get("X_TEST_KEY", None)
Y_TEST = os.environ.get("Y_TEST_KEY", None)

if not all([X_TRAIN, Y_TRAIN, X_TEST, Y_TEST]):
    print("Error: Not all data file keys (X_TRAIN, Y_TRAIN, X_TEST, Y_TEST) are provided as environment variables.")
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


# Load your data
X = load_data_from_minio(X_TRAIN)
y = load_data_from_minio(Y_TRAIN)
X_test = load_data_from_minio(X_TEST)
y_test = load_data_from_minio(Y_TEST)

if X is None or y is None or X_test is None or y_test is None:
    print("Failed to load all necessary data. Exiting.")
    exit(1)

# Configure MLflow
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME"))

# mlflow.create_experiment(
    # name=os.environ.get("MLFLOW_EXPERIMENT_NAME"),
    # artifact_location="s3://mlartifacts",
# )


# Start MLflow run
with mlflow.start_run():
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("C", C)
    mlflow.log_param("random_state", random_state)
    mlflow.log_param("cv", cv)

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    accuracy_scores = []
    f1_scores = []

    for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Train the model
        model = LogisticRegression(C=C, random_state=random_state)
        model.fit(X_train, y_train)

        # Make predictions on validation set
        y_pred = model.predict(X_val)

        # Evaluate the model
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average="weighted")  # Use weighted average for multi-class
        accuracy_scores.append(accuracy)
        f1_scores.append(f1)

        # Log metrics for each fold
        mlflow.log_metric(f"accuracy_fold_{fold}", accuracy)
        mlflow.log_metric(f"f1_fold_{fold}", f1)

    # Calculate and log average metrics
    avg_accuracy = np.mean(accuracy_scores)
    avg_f1 = np.mean(f1_scores)
    mlflow.log_metric("accuracy_avg", avg_accuracy)
    mlflow.log_metric("f1_avg", avg_f1)

    # Train final model on the entire training set
    final_model = LogisticRegression(C=C, random_state=random_state)
    final_model.fit(X, y)

    # Evaluate on the training set
    y_pred_train = final_model.predict(X)
    train_accuracy = accuracy_score(y, y_pred_train)
    train_f1 = f1_score(y, y_pred_train, average="weighted")
    mlflow.log_metric("train_accuracy", train_accuracy)
    mlflow.log_metric("train_f1", train_f1)

    # Evaluate on the test set
    y_pred_test = final_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test, average="weighted")
    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_metric("test_f1", test_f1)


    # Log the final model
    mlflow.sklearn.log_model(final_model, "model")

    print(f"model saved to: {mlflow.get_artifact_uri('model')}")
