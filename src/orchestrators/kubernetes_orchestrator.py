import subprocess
import os
import yaml
from itertools import product
from dotenv import load_dotenv
import multiprocessing

load_dotenv("s3.env")
load_dotenv("mlflow.env")

# Define hyperparameter ranges
C_values = [0.1, 0.5, 1.0]
random_state_values = [42]
cv_values = [5]

template_file = "./templates/job_template.yaml"
with open(template_file, "r") as f:
    job_template = f.read()

# Function to generate hyperparameter combinations
def generate_hyperparameter_combinations():
    return product(C_values, random_state_values, cv_values)

def deploy_job(C, random_state, cv):
    job_name = f"ml-training-job-{C}-{random_state}"
    job_file = f"ml_training_job_{C}_{random_state}.yaml"

    # Create job definition
    job_definition = job_template.format(
        C=C,
        random_state=random_state,
        cv=cv,
        minio_endpoint=os.environ.get("MINIO_ENDPOINT"),
        minio_access_key=os.environ.get("MINIO_ACCESS_KEY"),
        minio_secret_key=os.environ.get("MINIO_SECRET_KEY"),
        minio_bucket=os.environ.get("MINIO_BUCKET"),
        mlflow_tracking_uri=os.environ.get("MLFLOW_TRACKING_URI"),
        mlflow_experiment_name=os.environ.get("MLFLOW_EXPERIMENT_NAME"),
        mlflow_s3_endpoint_url=os.environ.get("MLFLOW_S3_ENDPOINT_URL")
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
        print(f"Job {job_name} deployed successfully with C={C} and RANDOM_STATE={random_state}.")

    except subprocess.CalledProcessError as e:
        print(f"Error deploying job {job_name} with C={C} and RANDOM_STATE={random_state}: {e.stderr}")

    finally:
        # Clean up job definition file
        os.remove(job_file)

if __name__ == '__main__':
    # Create a pool of processes
    pool = multiprocessing.Pool()

    # Generate hyperparameter combinations
    hyperparameter_combinations = generate_hyperparameter_combinations()

    # Deploy jobs in parallel
    pool.starmap(deploy_job, hyperparameter_combinations)

    # Close the pool and wait for all processes to complete
    pool.close()
    pool.join()

    print("All hyperparameter combinations processed.")
