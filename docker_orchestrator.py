import subprocess
import os
import dotenv

# Execute dvc status and capture the output
try:
    result = subprocess.run(
        ["dvc", "status", "--show-hash"], capture_output=True, text=True, check=True
    )
    dvc_status = result.stdout
except subprocess.CalledProcessError as e:
    print(f"Error running dvc status: {e}")
    exit(1)

# Parse the output to extract the hashes
x_train_hash = None
x_test_hash = None
y_train_hash = None
y_test_hash = None

for line in dvc_status.splitlines():
    if "X_train.pkl.dvc:" in line:
        x_train_hash = line.split()[1]
    if "X_test.pkl.dvc:" in line:
        x_test_hash = line.split()[1]
    if "y_train.pkl.dvc:" in line:
        y_train_hash = line.split()[1]
    if "y_test.pkl.dvc:" in line:
        y_test_hash = line.split()[1]

# Create a .env file with the data hashes
if (
    x_train_hash is None
    or x_test_hash is None
    or y_train_hash is None
    or y_test_hash is None
):
    print("Error: Could not extract all data hashes from dvc status output.")
    exit(1)

# Write hashes to .env file
with open(".env", "w") as f:
    f.write(f"X_TRAIN_HASH={x_train_hash}\n")
    f.write(f"X_TEST_HASH={x_test_hash}\n")
    f.write(f"Y_TRAIN_HASH={y_train_hash}\n")
    f.write(f"Y_TEST_HASH={y_test_hash}\n")

# Run the Docker container with the .env file
try:
    subprocess.run(
        [
            "docker",
            "run",
            "--env-file",
            ".env",
            "--env",
            "C=0.5",
            "--env",
            "RANDOM_STATE=123",
            "your_image_name",
        ],
        check=True,
    )
    print("Docker container ran successfully.")
except subprocess.CalledProcessError as e:
    print(f"Error running docker container: {e}")
    exit(1)

# Clean up the .env file
os.remove(".env")
