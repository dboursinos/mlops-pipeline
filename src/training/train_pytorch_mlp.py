import os
import mlflow
import mlflow.pytorch
import boto3
from dotenv import load_dotenv
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
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

# PyTorch MLP specific hyperparameters
num_epochs = int(os.environ.get("NUM_EPOCHS", 10))
learning_rate = float(os.environ.get("LEARNING_RATE", 0.001))
batch_size = int(os.environ.get("BATCH_SIZE", 32))
hidden_layer_sizes_str = os.environ.get("HIDDEN_LAYER_SIZES", "64,32")
# Parse hidden layer sizes from comma-separated string
hidden_layer_sizes = [int(x.strip()) for x in hidden_layer_sizes_str.split(',') if x.strip().isdigit()]

# Determine device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# Set manual seed for reproducibility on CPU and GPU
torch.manual_seed(common_random_state)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(common_random_state)


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

input_size = X.shape[1]
num_classes = len(np.unique(y))
print(f"Inferred input_size: {input_size}, num_classes: {num_classes}")

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, num_classes):
        super(SimpleMLP, self).__init__()
        layers = []
        # Input layer to first hidden layer
        layers.append(nn.Linear(input_size, hidden_layer_sizes[0]))
        layers.append(nn.ReLU()) # Using ReLU activation

        # Hidden layers
        for i in range(len(hidden_layer_sizes) - 1):
            layers.append(nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i+1]))
            layers.append(nn.ReLU())

        # Last hidden layer to output layer
        layers.append(nn.Linear(hidden_layer_sizes[-1], num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# --- MLflow Setup ---
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME"))

# --- Main Training Logic ---
with mlflow.start_run():
    mlflow.log_param("model_type", "PyTorchMLP")
    mlflow.log_param("random_state", common_random_state)
    mlflow.log_param("cv", common_cv)

    # Log PyTorch MLP specific hyperparameters
    mlflow.log_param("num_epochs", num_epochs)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("input_size", input_size)
    mlflow.log_param("num_classes", num_classes)

    print(f"Training PyTorch MLP with {num_epochs} epochs, LR={learning_rate}, Batch Size={batch_size}, Hidden Layers={hidden_layer_sizes}...")

    skf = StratifiedKFold(n_splits=common_cv, shuffle=True, random_state=common_random_state)

    accuracy_scores = []
    f1_scores = []

    # Perform cross-validation
    for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
        X_train_fold, X_val_fold = X[train_index], X[val_index]
        y_train_fold, y_val_fold = y[train_index], y[val_index]

        # Convert fold data to PyTorch tensors and move to device
        X_train_tensor = torch.tensor(X_train_fold, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train_fold, dtype=torch.long).to(device)
        X_val_tensor = torch.tensor(X_val_fold, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val_fold, dtype=torch.long).to(device)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Instantiate model, loss, and optimizer for each fold
        model = SimpleMLP(input_size, hidden_layer_sizes, num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop for the current fold
        for epoch in range(num_epochs):
            model.train() # Set model to training mode
            running_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(train_dataset)
            mlflow.log_metric(f"train_loss_fold_{fold}_epoch_{epoch}", epoch_loss)

        # Evaluate model on validation set for the current fold
        model.eval() # Set model to evaluation mode
        with torch.no_grad(): # Disable gradient calculation for evaluation
            outputs = model(X_val_tensor)
            _, predicted = torch.max(outputs.data, 1)

            accuracy = accuracy_score(y_val_tensor.cpu().numpy(), predicted.cpu().numpy())
            f1 = f1_score(y_val_tensor.cpu().numpy(), predicted.cpu().numpy(), average="weighted")

            accuracy_scores.append(accuracy)
            f1_scores.append(f1)

            mlflow.log_metric(f"accuracy_fold_{fold}", accuracy)
            mlflow.log_metric(f"f1_fold_{fold}", f1)

    avg_accuracy = np.mean(accuracy_scores)
    avg_f1 = np.mean(f1_scores)
    mlflow.log_metric("accuracy_avg", avg_accuracy)
    mlflow.log_metric("f1_avg", avg_f1)

    # Train final model on the entire training set (X, y)
    print("Training final model on full dataset...")
    final_model = SimpleMLP(input_size, hidden_layer_sizes, num_classes).to(device)
    final_criterion = nn.CrossEntropyLoss()
    final_optimizer = optim.Adam(final_model.parameters(), lr=learning_rate)

    # Convert full training data to tensors
    X_full_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_full_tensor = torch.tensor(y, dtype=torch.long).to(device)
    full_train_dataset = TensorDataset(X_full_tensor, y_full_tensor)
    full_train_loader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs): # Train final model for the same number of epochs
        final_model.train()
        for inputs, labels in full_train_loader:
            final_optimizer.zero_grad()
            outputs = final_model(inputs)
            loss = final_criterion(outputs, labels)
            loss.backward()
            final_optimizer.step()

    # Evaluate on the training set
    final_model.eval()
    with torch.no_grad():
        outputs_train = final_model(X_full_tensor)
        _, predicted_train = torch.max(outputs_train.data, 1)
        train_accuracy = accuracy_score(y_full_tensor.cpu().numpy(), predicted_train.cpu().numpy())
        train_f1 = f1_score(y_full_tensor.cpu().numpy(), predicted_train.cpu().numpy(), average="weighted")
    mlflow.log_metric("train_accuracy", train_accuracy)
    mlflow.log_metric("train_f1", train_f1)

    # Evaluate on the test set
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
    with torch.no_grad():
        outputs_test = final_model(X_test_tensor)
        _, predicted_test = torch.max(outputs_test.data, 1)
        test_accuracy = accuracy_score(y_test_tensor.cpu().numpy(), predicted_test.cpu().numpy())
        test_f1 = f1_score(y_test_tensor.cpu().numpy(), predicted_test.cpu().numpy(), average="weighted")
    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_metric("test_f1", test_f1)

    # Log the final model
    mlflow.pytorch.log_model(final_model, "model")

    print(f"model saved to: {mlflow.get_artifact_uri('model')}")
    print(f"Finished training PyTorch MLP with common_params: {{'RANDOM_STATE': {common_random_state}, 'CV': {common_cv}}}, model_specific_params: {{'NUM_EPOCHS': {num_epochs}, 'LEARNING_RATE': {learning_rate}, 'BATCH_SIZE': {batch_size}, 'HIDDEN_LAYER_SIZES': '{hidden_layer_sizes_str}'}}")
