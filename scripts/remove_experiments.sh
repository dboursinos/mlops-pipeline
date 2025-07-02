#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
# Define paths to your .env files relative to this script's location
# Assuming the scripts directory is at the same level as your .env files
# e.g., if .env files are in the project root and this script is in 'scripts/'
ENV_DIR="../" # Adjust this path if your .env files are elsewhere

DB_ENV_FILE="${ENV_DIR}db.env"
MLFLOW_ENV_FILE="${ENV_DIR}mlflow.env"
S3_ENV_FILE="${ENV_DIR}s3.env"

DOCKER_CONTAINER_NAME="mlflow_server"

# --- Load Environment Variables ---
# Use 'set -a' to automatically export variables defined after it
# Use 'source' to load the variables into the current shell
echo "Loading environment variables from $DB_ENV_FILE, $MLFLOW_ENV_FILE, $S3_ENV_FILE..."
set -a # Automatically export all subsequent variables
source "$DB_ENV_FILE"
source "$MLFLOW_ENV_FILE"
source "$S3_ENV_FILE"
set +a # Turn off automatic export

# --- Validate Required Variables ---
: "${POSTGRES_USER?Error: POSTGRES_USER not set in db.env}"
: "${POSTGRES_PASSWORD?Error: POSTGRES_PASSWORD not set in db.env}"
: "${POSTGRES_HOST?Error: POSTGRES_HOST not set in db.env}"
: "${POSTGRES_PORT?Error: POSTGRES_PORT not set in db.env}"
: "${POSTGRES_DB?Error: POSTGRES_DB not set in db.env}"
: "${MINIO_BUCKET?Error: MINIO_BUCKET not set in s3.env}"

# --- Construct MLflow URIs ---
BACKEND_STORE_URI="postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DB}"
ARTIFACTS_DESTINATION="s3://${MINIO_BUCKET}"

echo "Executing MLflow GC command..."
echo "Backend Store URI: ${BACKEND_STORE_URI}"
echo "Artifacts Destination: ${ARTIFACTS_DESTINATION}"

# --- Execute MLflow GC Command inside Docker Container ---
docker exec -it \
  --env-file "$DB_ENV_FILE" \
  --env-file "$MLFLOW_ENV_FILE" \
  --env-file "$S3_ENV_FILE" \
  "${DOCKER_CONTAINER_NAME}" \
  mlflow gc \
  --backend-store-uri "${BACKEND_STORE_URI}" \
  --artifacts-destination "${ARTIFACTS_DESTINATION}"
