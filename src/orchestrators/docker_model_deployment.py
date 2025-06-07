import subprocess
from dotenv import load_dotenv

load_dotenv("s3.env")
load_dotenv("mlflow.env")
load_dotenv("prod.env")

def run_docker_container():
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
                "prod.env",
                "--network",
                "host",
                "ml_production",
            ],
            check=True,
        )
        print("Model deployed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error running docker container: {e}")
        return 1
    return 0


if __name__ == "__main__":
    run_docker_container()
