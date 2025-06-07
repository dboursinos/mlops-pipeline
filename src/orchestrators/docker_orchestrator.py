import subprocess
import os
from itertools import product
import multiprocessing

# Define hyperparameters range
C_values = [0.1, 0.5, 1.0]
random_state_values = [42, 123]
cv_values = [5]

def run_docker_container(C, random_state, cv):
    env_file = f".env_{C}_{random_state}"
    with open(env_file, "w") as f:
        f.write(f"C={C}\n")
        f.write(f"RANDOM_STATE={random_state}\n")
        f.write(f"CV={cv}\n")

    try:
        subprocess.run(
            [
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
                "ml_training",
            ],
            check=True,
        )
        print(f"Docker container ran successfully with C={C} and RANDOM_STATE={random_state}.")
    except subprocess.CalledProcessError as e:
        print(f"Error running docker container with C={C} and RANDOM_STATE={random_state}: {e}")
        return 1
    finally:
        os.remove(env_file)
    return 0

if __name__ == "__main__":
    # Hyperparameter generator
    hyperparameters = product(C_values, random_state_values, cv_values)

    # Create a pool of processes
    pool = multiprocessing.Pool()

    # Run docker containers in parallel
    results = pool.starmap(run_docker_container, hyperparameters)

    # Close the pool and wait for all processes to complete
    pool.close()
    pool.join()

    if any(results):  # Check if any of the runs failed
        print("One or more docker containers failed to run.")
        exit(1)
    else:
        print("All docker containers ran successfully.")
