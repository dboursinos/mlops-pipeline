import subprocess
import os
from itertools import product
import multiprocessing
import yaml

try:
    with open("./src/config/training_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    hyperparameter_ranges = config.get("hyperparameters", {})
    image_name = config.get("image")
    if not image_name:
        raise ValueError("Image name not found in training_config.yaml. Please specify 'image'.")
except FileNotFoundError:
    print("Error: training_config.yaml not found. Please create it with hyperparameter and image definitions.")
    exit(1)
except yaml.YAMLError as e:
    print(f"Error parsing training_config.yaml: {e}")
    exit(1)
except ValueError as e:
    print(f"Configuration Error: {e}")
    exit(1)

def generate_hyperparameter_combinations():
    keys = hyperparameter_ranges.keys()
    values = hyperparameter_ranges.values()

    all_combinations_values = product(*values)

    for combination_tuple in all_combinations_values:
        yield dict(zip(keys, combination_tuple))

def run_docker_container(hyperparams: dict):
    # Generate a unique name for the env file based on hyperparameters
    # Sort keys for consistent naming and replace '.' for valid file name
    param_parts = [f"{k}-{str(v).replace('.', 'p')}" for k, v in sorted(hyperparams.items())]
    env_file_suffix = "_".join(param_parts)
    env_file = f".env_run_{env_file_suffix}"

    with open(env_file, "w") as f:
        for key, value in hyperparams.items():
            f.write(f"{key}={value}\n")

    try:
        command = [
            "docker",
            "run",
            "--rm",
            "--env-file",
            "s3.env",
            "--env-file",
            "mlflow.env",
            "--env-file",
            env_file,
            "--network",
            "host",
            image_name,
        ]
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"Docker container ran successfully with hyperparameters: {hyperparams}.")
    except subprocess.CalledProcessError as e:
        print(f"Error running docker container with hyperparameters {hyperparams}: {e.stderr}")
        return 1
    finally:
        os.remove(env_file)
    return 0

if __name__ == "__main__":
    # Hyperparameter generator
    hyperparameter_combinations = list(generate_hyperparameter_combinations())

    # Create a pool of processes
    pool = multiprocessing.Pool()

    # Run docker containers in parallel
    results = pool.map(run_docker_container, hyperparameter_combinations)

    # Close the pool and wait for all processes to complete
    pool.close()
    pool.join()

    if any(results):  # Check if any of the runs failed
        print("One or more docker containers failed to run.")
        exit(1)
    else:
        print("All docker containers ran successfully.")
