import subprocess
import os
import yaml
from itertools import product
from dotenv import load_dotenv
import multiprocessing

load_dotenv("s3.env")
load_dotenv("mlflow.env")

# Load hyperparameter ranges from config.yaml
try:
    with open("./src/training/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    hyperparameter_ranges = config.get("hyperparameters", {})
    image_name = config.get("image")
    if not image_name:
        raise ValueError("Image name not found in config.yaml. Please specify 'image'.")
except FileNotFoundError:
    print("Error: config.yaml not found. Please create it with hyperparameter definitions.")
    exit(1)
except yaml.YAMLError as e:
    print(f"Error parsing config.yaml: {e}")
    exit(1)


template_file = "./templates/job_template.yaml"
with open(template_file, "r") as f:
    job_template = f.read()

# Function to generate hyperparameter combinations
def generate_hyperparameter_combinations():
    keys = hyperparameter_ranges.keys()
    values = hyperparameter_ranges.values()

    all_combinations_values = product(*values)

    # Map each combination of values back to a dictionary with original keys
    for combination in all_combinations_values:
        yield dict(zip(keys, combination))

def deploy_job(hyperparams: dict):
    # Generate a unique name for the job based on hyperparameters
    # Sort keys for consistent naming and replace '.' for valid Kubernetes name
    param_parts = [f"{k.lower().replace('_', '-')}-{str(v).replace('.', 'p')}" for k, v in sorted(hyperparams.items())]
    job_name_suffix = "-".join(param_parts)
    job_name = f"ml-training-job-{job_name_suffix}"
    job_file = f"ml_training_job_{job_name_suffix}.yaml"

    # Generate environment variables block for the YAML template
    env_vars_yaml = []
    for key, value in hyperparams.items():
        env_vars_yaml.append(f"        - name: {key}")
        env_vars_yaml.append(f"          value: \"{value}\"")

    # Add existing environment variables from .env files
    existing_env_vars = {
        "MINIO_ENDPOINT": os.environ.get("MINIO_ENDPOINT"),
        "MINIO_ACCESS_KEY": os.environ.get("MINIO_ACCESS_KEY"),
        "MINIO_SECRET_KEY": os.environ.get("MINIO_SECRET_KEY"),
        "MINIO_BUCKET": os.environ.get("MINIO_BUCKET"),
        "MLFLOW_TRACKING_URI": os.environ.get("MLFLOW_TRACKING_URI"),
        "MLFLOW_EXPERIMENT_NAME": os.environ.get("MLFLOW_EXPERIMENT_NAME"),
        "MLFLOW_S3_ENDPOINT_URL": os.environ.get("MLFLOW_S3_ENDPOINT_URL"),
        "AWS_ACCESS_KEY_ID": os.environ.get("MINIO_ACCESS_KEY"), # For S3 client compatibility
        "AWS_SECRET_ACCESS_KEY": os.environ.get("MINIO_SECRET_KEY") # For S3 client compatibility
    }
    for key, value in existing_env_vars.items():
        if value is not None:
            env_vars_yaml.append(f"        - name: {key}")
            env_vars_yaml.append(f"          value: \"{value}\"")

    dynamic_env_vars_string = "\n".join(env_vars_yaml)

    # Create job definition
    job_definition = job_template.format(
        job_name=job_name,
        image_name=image_name,
        dynamic_env_vars=dynamic_env_vars_string
    )

    with open(job_file, "w") as f:
        f.write(job_definition)

    try:
        # Deploy job to Kubernetes
        subprocess.run(
            ["kubectl", "apply", "-f", job_file],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"Job {job_name} deployed successfully with hyperparameters: {hyperparams}.")

    except subprocess.CalledProcessError as e:
        print(f"Error deploying job {job_name} with hyperparameters {hyperparams}: {e.stderr}")

    finally:
        # Clean up job definition file
        os.remove(job_file)

if __name__ == '__main__':
    # Create a pool of processes
    pool = multiprocessing.Pool()

    # Generate hyperparameter combinations
    hyperparameter_combinations = list(generate_hyperparameter_combinations())

    # Deploy jobs in parallel
    pool.map(deploy_job, hyperparameter_combinations)

    # Close the pool and wait for all processes to complete
    pool.close()
    pool.join()

    print("All hyperparameter combinations processed.")
