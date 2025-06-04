# Run using docker run --env C=0.5 --env RANDOM_STATE=123 your_image_name


import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Define hyperparameters with default values
C = float(os.environ.get("C", 1.0))  # Regularization strength
random_state = int(
    os.environ.get("RANDOM_STATE", 42)
)  # Random state for reproducibility

# Generate artificial data
X, y = make_classification(n_samples=1000, n_features=20, random_state=random_state)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_state
)

# Start MLflow run
with mlflow.start_run():
    # Log hyperparameters
    mlflow.log_param("C", C)
    mlflow.log_param("random_state", random_state)

    # Train the model
    model = LogisticRegression(C=C, random_state=random_state)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", accuracy)

    # Log the model
    mlflow.sklearn.log_model(model, "model")

    print(f"Accuracy: {accuracy}")
    print(f"Model saved to: {mlflow.get_artifact_uri('model')}")
