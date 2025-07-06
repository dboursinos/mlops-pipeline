import subprocess
import os
import yaml
from itertools import product
from dotenv import load_dotenv
import hashlib
import time
from collections import deque
import json
from tqdm import tqdm

load_dotenv("s3.env")
load_dotenv("mlflow.env")

try:
    with open("./src/config/training_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("Config file 'training_config.yaml' must contain a dictionary at its root.")

    experiment_name = config.get("experiment_name", "default_experiment")
    data_files = config.get("data_files", {})
    models_config = config.get("models", [])
    kubernetes_config = config.get("kubernetes_config", {})

    if not isinstance(experiment_name, str):
        raise ValueError("The 'experiment_name' section in training_config.yaml must be a string.")
    if not isinstance(data_files, dict):
        raise ValueError("The 'data_files' section in training_config.yaml must be a dictionary.")
    if not isinstance(models_config, list):
        raise ValueError("The 'models' section in training_config.yaml must be a list of model definitions.")
    if not models_config:
        raise ValueError("No models defined in training_config.yaml under 'models' section.")
    if not isinstance(kubernetes_config, dict):
        raise ValueError("The 'kubernetes_config' section in training_config.yaml must be a dictionary.")
    K8S_NAMESPACE = kubernetes_config.get("namespace", "default")
    MAX_CONCURRENT_JOBS = kubernetes_config.get("max_concurrent_jobs", 3)
    POLLING_INTERVAL_SECONDS = kubernetes_config.get("polling_interval_seconds", 15)

except FileNotFoundError:
    print("Error: training_config.yaml not found. Please create it with model and hyperparameter definitions.")
    exit(1)
except yaml.YAMLError as e:
    print(f"Error parsing training_config.yaml: {e}")
    exit(1)
except ValueError as e:
    print(f"Configuration Error: {e}")
    exit(1)

try:
    subprocess.run(
        ["kubectl", "create", "namespace", K8S_NAMESPACE],
        check=False,  # Don't raise an error if the namespace already exists
        capture_output=True,
        text=True,
    )
    print(f"Namespace {K8S_NAMESPACE} created (if it didn't exist).")
except subprocess.CalledProcessError as e:
    print(f"Error creating namespace: {e.stderr}")


template_file = "./templates/job_template.yaml"
try:
    with open(template_file, "r") as f:
        job_template = f.read()
except FileNotFoundError:
    print(f"Error: Job template file '{template_file}' not found.")
    exit(1)
except Exception as e:
    print(f"Error reading job template file '{template_file}': {e}")
    exit(1)

# Function to generate hyperparameter combinations
def generate_all_combinations():
    all_model_combinations = []
    for model_def in models_config:
        model_name = model_def.get("name")
        model_image = model_def.get("image")
        model_hyperparameters = model_def.get("hyperparameters", {})

        if not model_name:
            raise ValueError("Each model in 'models' list must have a 'name'.")
        if not model_image:
            raise ValueError(f"Model '{model_name}' must have an 'image' defined.")
        if not isinstance(model_hyperparameters, dict):
            raise ValueError(f"Hyperparameters for model '{model_name}' must be a dictionary.")

        keys = model_hyperparameters.keys()
        values = model_hyperparameters.values()

        # Process values to ensure they are always iterables (e.g., lists),
        # even if a hyperparameter is defined as a single scalar.
        # This prevents 'itertools.product' from iterating over characters of a string.
        processed_values = []
        for v in values:
            if isinstance(v, (list, tuple)):
                processed_values.append(v)
            else:
                processed_values.append([v])

        # Generate combinations for the current model
        for combo_values in product(*processed_values):
            combo_dict = dict(zip(keys, combo_values))
            combo_dict["MODEL_TYPE"] = model_name
            combo_dict["DOCKER_IMAGE"] = model_image
            all_model_combinations.append(combo_dict)
    return all_model_combinations

def _get_active_kubernetes_jobs(namespace: str, label_selector: str) -> int:
    """
    Returns the count of active Kubernetes Jobs matching the label selector using kubectl.
    """
    try:
        cmd = [
            "kubectl", "get", "jobs",
            "-n", namespace,
            "-l", label_selector,
            "-o", "json"
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        jobs_json = json.loads(result.stdout)

        active_count = 0
        for job in jobs_json.get("items", []):
            status = job.get("status", {})
            # A job is considered active if its 'active' count in status is greater than 0
            if status.get("active", 0) > 0:
                active_count += 1
        return active_count
    except subprocess.CalledProcessError as e:
        tqdm.write(f"Error executing kubectl command: {e.stderr}")
        return 0 # Return 0 active jobs on error
    except json.JSONDecodeError as e:
        tqdm.write(f"Error parsing kubectl JSON output: {e}")
        return 0
    except Exception as e:
        tqdm.write(f"Unexpected error fetching Kubernetes jobs: {e}")
        return 0

def deploy_job(job_params: dict):
    model_type = job_params.get("MODEL_TYPE", "unknown-model")
    image_name_for_job = job_params.get("DOCKER_IMAGE")

    if not image_name_for_job:
        tqdm.write(f"Error: No Docker image specified for model type {model_type}. Skipping deployment.")
        return

    params_for_naming = {k: v for k, v in job_params.items() if k not in ["MODEL_TYPE", "DOCKER_IMAGE"]}

    param_parts = [
        f"{k.lower().replace('_', '-')}-{str(v).replace('.', 'p')}"
        for k, v in sorted(params_for_naming.items())
    ]
    job_name_base = f"ml-{model_type.lower().replace('_', '-')}"
    job_name_suffix = "-".join(param_parts)

    full_job_name = f"{job_name_base}-{job_name_suffix}"

    # Kubernetes job names have a max length of 63 characters.
    # If it's too long, truncate and add a hash for uniqueness.
    if len(full_job_name) > 63:
        # Calculate a hash of the full suffix for uniqueness
        short_hash = hashlib.sha1(job_name_suffix.encode()).hexdigest()[:8]
        # Construct a shorter name: prefix-model-hash
        job_name = f"ml-{model_type.lower().replace('_', '-')}-{short_hash}"
        # tqdm.write(f"Warning: Original job name '{full_job_name}' was too long. Truncated to '{job_name}'")
    else:
        job_name = full_job_name

    job_file = f"ml_training_job_{job_name}.yaml"

    # tqdm.write(f"Attempting to deploy job: {job_name} with model {model_type}")

    # Generate environment variables block for the YAML template
    env_vars_yaml = []
    # Add all parameters from the combined dictionary as environment variables, including MODEL_TYPE
    for key, value in job_params.items():
        # Do not add DOCKER_IMAGE as an environment variable to the training script
        if key == "DOCKER_IMAGE":
            continue
        env_vars_yaml.append(f"        - name: {key}")
        env_vars_yaml.append(f"          value: \"{str(value)}\"")

    # Add data file paths as environment variables
    for key, value in data_files.items():
        if value is not None:
            env_vars_yaml.append(f"        - name: {key}")
            env_vars_yaml.append(f"          value: \"{value}\"")

    # Add existing environment variables from .env files
    existing_env_vars = {
        "MINIO_ENDPOINT": os.environ.get("MINIO_ENDPOINT"),
        "MINIO_ACCESS_KEY": os.environ.get("MINIO_ACCESS_KEY"),
        "MINIO_SECRET_KEY": os.environ.get("MINIO_SECRET_KEY"),
        "MINIO_BUCKET": os.environ.get("MINIO_BUCKET"),
        "MLFLOW_TRACKING_URI": os.environ.get("MLFLOW_TRACKING_URI"),
        "MLFLOW_EXPERIMENT_NAME": experiment_name,
        "MLFLOW_S3_ENDPOINT_URL": os.environ.get("MLFLOW_S3_ENDPOINT_URL"),
        "AWS_ACCESS_KEY_ID": os.environ.get("MINIO_ACCESS_KEY"),
        "AWS_SECRET_ACCESS_KEY": os.environ.get("MINIO_SECRET_KEY")
    }
    for key, value in existing_env_vars.items():
        if value is not None:
            env_vars_yaml.append(f"        - name: {key}")
            env_vars_yaml.append(f"          value: \"{value}\"")

    dynamic_env_vars_string = "\n".join(env_vars_yaml)

    # Create job definition
    job_definition = job_template.format(
        job_name=job_name,
        image_name=image_name_for_job,
        dynamic_env_vars=dynamic_env_vars_string
    )

    with open(job_file, "w") as f:
        f.write(job_definition)

    try:
        subprocess.run(
            ["kubectl", "apply", "-f", job_file, "-n", K8S_NAMESPACE],
            check=True,
            capture_output=True,
            text=True
        )
        # tqdm.write(f"Job {job_name} deployed successfully with model {model_type} and hyperparameters: {job_params}.")

    except subprocess.CalledProcessError as e:
        tqdm.write(f"Error deploying job {job_name} with model {model_type} and hyperparameters {job_params}: {e.stderr}")

    finally:
        os.remove(job_file)

if __name__ == '__main__':
    all_job_combinations = generate_all_combinations()
    pending_jobs = deque(all_job_combinations)

    total_jobs_to_schedule = len(pending_jobs)

    print(f"Experiment Name: {experiment_name}")
    print(f"Total jobs to schedule: {total_jobs_to_schedule}")
    print(f"Maximum concurrent jobs: {MAX_CONCURRENT_JOBS}")
    print(f"Kubernetes Namespace: {K8S_NAMESPACE}")
    print(f"Polling interval: {POLLING_INTERVAL_SECONDS} seconds")

    # Initialize tqdm progress bar with custom format to remove incorrect estimated remaining time
    # {desc}: description (from set_description)
    # {percentage:.0f}%: current percentage without decimals
    # {bar}: the actual progress bar graphic
    # {n_fmt}/{total_fmt}: current count / total count
    # {unit}: the unit, e.g., "job"
    custom_bar_format = "{desc}: {percentage:.0f}%|{bar}| {n_fmt}/{total_fmt} {unit}"

    # Initialize tqdm progress bar
    # desc: description, total: total iterations, unit: unit name, dynamic_ncols: adjust width
    # leave=True: keep bar on screen after completion
    with tqdm(total=total_jobs_to_schedule, desc="Scheduling Jobs", unit="jobs", dynamic_ncols=True, leave=True, bar_format=custom_bar_format) as pbar:
        # Initial active jobs check for the loop condition
        active_jobs = _get_active_kubernetes_jobs(K8S_NAMESPACE, 'app=ml-training-job')

        while pending_jobs or active_jobs > 0:
            active_jobs = _get_active_kubernetes_jobs(K8S_NAMESPACE, 'app=ml-training-job')

            pbar.set_description(f"Scheduling Jobs (Active: {active_jobs}/{MAX_CONCURRENT_JOBS}, Pending: {len(pending_jobs)})")

            if active_jobs < MAX_CONCURRENT_JOBS and pending_jobs:
                job_to_deploy = pending_jobs.popleft()
                deploy_job(job_to_deploy)
                pbar.update(1)
                time.sleep(1)
            else:
                if not pending_jobs and active_jobs == 0:
                    pbar.set_description("All jobs completed.")
                    break # All jobs processed and no active jobs remaining

                # tqdm.write(f"Max concurrent jobs reached or no jobs to deploy. Waiting for {POLLING_INTERVAL_SECONDS} seconds...")
                time.sleep(POLLING_INTERVAL_SECONDS)

    tqdm.write("\nAll training jobs completed.")
