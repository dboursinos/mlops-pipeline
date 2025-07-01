import subprocess
import os
from dotenv import load_dotenv
import yaml

load_dotenv("s3.env")
load_dotenv("mlflow.env")

def load_deployment_config(config_path: str = "./src/config/model_deployment_config.yaml"):
    """Loads model deployment configuration from a YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Deployment configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration file {config_path}: {e}")

def deploy_model():
    """Deploys the MLflow model to Kubernetes using provided templates."""

    deployment_config = load_deployment_config()
    mlflow_model_uri = deployment_config.get("model_uri")
    replicas = deployment_config.get("replicas", 1)

    if not mlflow_model_uri:
        raise ValueError("Error: 'model_uri' must be specified in model_deployment_config.yaml")
    if not isinstance(replicas, int) or replicas < 1:
        raise ValueError("Error: 'replicas' must be a positive integer in model_deployment_config.yaml")

    with open("templates/production_deployment_template.yaml", "r") as f:
        deployment_template = f.read()
    with open("templates/production_service_template.yaml", "r") as f:
        service_template = f.read()
    with open("templates/production_ingress_template.yaml", "r") as f:
        ingress_template = f.read()

    # Fill in the templates with the provided information
    deployment_definition = deployment_template.format(
        replicas=replicas,
        mlflow_model_uri=mlflow_model_uri,
        mlflow_tracking_uri=os.environ.get("MLFLOW_TRACKING_URI"),
        mlflow_s3_endpoint_url=os.environ.get("MLFLOW_S3_ENDPOINT_URL"),
        mlflow_s3_ignore_tls=os.environ.get("MLFLOW_S3_IGNORE_TLS"),
        minio_endpoint=os.environ.get("MINIO_ENDPOINT"),
        minio_access_key=os.environ.get("MINIO_ACCESS_KEY"),
        minio_secret_key=os.environ.get("MINIO_SECRET_KEY"),
    )
    # ingress_definition = ingress_template.format(
        # model_name=os.environ.get("MLFLOW_MODEL_URI").split("/")[-1],
    # )
    ingress_definition = ingress_template
    service_definition = service_template

    # Save the filled-in definitions to temporary files
    deployment_file = "mlflow_deployment.yaml"
    ingress_file = "mlflow_ingress.yaml"
    service_file = "mlflow_service.yaml"

    with open(deployment_file, "w") as f:
        f.write(deployment_definition)
    with open(ingress_file, "w") as f:
        f.write(ingress_definition)
    with open(service_file, "w") as f:
        f.write(service_definition)

    try:
        subprocess.run(
            ["kubectl", "create", "namespace", "ml-prod"],
            check=False,  # Don't raise an error if the namespace already exists
            capture_output=True,
            text=True,
        )
        print("Namespace ml-prod created (if it didn't exist).")
    except subprocess.CalledProcessError as e:
        print(f"Error creating namespace: {e.stderr}")

    try:
        subprocess.run(
            ["kubectl", "apply", "-f", deployment_file, "-n", "ml-prod"],
            check=True,
            capture_output=True,
            text=True,
        )
        print("Deployment applied successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error applying deployment: {e.stderr}")

    try:
        subprocess.run(
            ["kubectl", "apply", "-f", ingress_file, "-n", "ml-prod"],
            check=True,
            capture_output=True,
            text=True,
        )
        print("Ingress applied successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error applying ingress: {e.stderr}")

    try:
        subprocess.run(
            ["kubectl", "apply", "-f", service_file, "-n", "ml-prod"],
            check=True,
            capture_output=True,
            text=True,
        )
        print("Service applied successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error applying service: {e.stderr}")

    finally:
        os.remove(deployment_file)
        os.remove(ingress_file)
        os.remove(service_file)


if __name__ == "__main__":
    deploy_model()
