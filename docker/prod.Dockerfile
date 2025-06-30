FROM python:3.13.4-slim

WORKDIR /app

RUN pip install mlflow==2.22.0 scikit-learn==1.6.1 boto3==1.26.131 xgboost==3.0.2 \
  && pip install --no-cache-dir torch==2.7.1+cpu -f https://download.pytorch.org/whl/torch

EXPOSE 5001

ENV MLFLOW_MODEL_URI=""
ENV MLFLOW_TRACKING_URI=""

# Serve the model using MLflow
ENTRYPOINT mlflow models serve --model-uri "$MLFLOW_MODEL_URI" --host 0.0.0.0 --port 5001 --env-manager=local
