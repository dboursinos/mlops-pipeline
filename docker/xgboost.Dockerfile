FROM python:3.13.4-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  libgomp1 \
  && rm -rf /var/lib/apt/lists/*

COPY ./docker/requirements/requirements.xgboost.txt .
RUN pip install --no-cache-dir -r requirements.xgboost.txt

RUN mkdir -p src/training/

COPY src/training/train_xgboost.py src/training/train_xgboost.py

# Command to run the training script
CMD ["python", "src/training/train_xgboost.py"]
