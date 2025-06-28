FROM python:3.13.4-slim

WORKDIR /app

COPY ./docker/requirements/requirements.pytorch.txt .
RUN pip install --no-cache-dir -r requirements.pytorch.txt \
  # Install PyTorch for CPU
  && pip install --no-cache-dir torch==2.7.1+cpu -f https://download.pytorch.org/whl/torch

# Create the directory structure for your script
RUN mkdir -p src/training/

# Copy the consolidated training script into the container
COPY ../src/training/train_pytorch_mlp.py src/training/train_pytorch_mlp.py

# Command to run the training script
CMD ["python", "src/training/train_pytorch_mlp.py"]
