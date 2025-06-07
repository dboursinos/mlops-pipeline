from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pickle
from dotenv import load_dotenv
import boto3
import os

load_dotenv(dotenv_path="s3.env")

aws_access_key_id = os.environ["MINIO_ACCESS_KEY"]
aws_secret_access_key = os.environ["MINIO_SECRET_KEY"]
aws_endpoint_url = os.environ["MINIO_ENDPOINT"]
aws_bucket = os.environ["MINIO_BUCKET"]
DATA_DIR = "data"

s3 = boto3.client(
    "s3",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    endpoint_url=aws_endpoint_url,
)

random_state = 42

X, y = make_classification(n_samples=1000, n_features=20, random_state=random_state)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_state
)

def save_data_locally(data, filename):
    """Saves data locally using pickle."""
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def save_data_to_minio(data, filename):
    """Saves data to MinIO using pickle."""
    try:
        # Serialize the data using pickle
        serialized_data = pickle.dumps(data)

        # Define the object key (path) in MinIO
        object_key = os.path.join(DATA_DIR, filename)

        # Upload the data to MinIO
        s3.put_object(Bucket=aws_bucket, Key=object_key, Body=serialized_data)

        print(f"Saved {filename} to MinIO bucket {aws_bucket} at {object_key}")

    except Exception as e:
        print(f"Error saving {filename} to MinIO: {e}")


save_data_locally(X_train, "./data/X_train.pkl")
save_data_locally(X_test, "./data/X_test.pkl")
save_data_locally(y_train, "./data/y_train.pkl")
save_data_locally(y_test, "./data/y_test.pkl")
save_data_to_minio(X_train, "X_train.pkl")
save_data_to_minio(X_test, "X_test.pkl")
save_data_to_minio(y_train, "y_train.pkl")
save_data_to_minio(y_test, "y_test.pkl")
