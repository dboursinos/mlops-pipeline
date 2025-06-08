# MLOps Pipeline for Scalable Model Training and Deployment

## Overview

A Docker and Kubernetes-ready MLOps pipeline for:

- Distributed hyperparameter tuning with cross-validation
- Model tracking via MLFlow (PostgreSQL + S3 backend)
- Scalable deployment options (Docker & Kubernetes)
- Production inference API with load balancing

## Prerequisites

- Docker
- Docker Compose
- Kubernetes cluster (optional)
- MLFlow tracking server
- MinIO (or S3-compatible storage)
- PostgreSQL

## Feature Architecture

```
Training Pipeline:
└── Hyperparameter Runs →
     ├── Docker containers (single node)
     └── Kubernetes pods (multi-node)
     └── MLFlow Tracking (PostgreSQL + MinIO)

Inference Pipeline:
└── MLFlow Model Serving →
     ├── Docker container
     └── Kubernetes replicas + Ingress
```

## Pipeline Structure

<img src="./images/pipeline.svg" width="100%" height="auto"/>

This diagram illustrates an end-to-end machine learning pipeline, from training to deployment.

1. **Training Phase:**
    - The pipeline begins with training data, model architecture, and hyperparameters as inputs to the Training Orchestrator.
    - The Training Orchestrator creates training jobs and distributes them either to a Kubernetes cluster with multiple nodes for distributed training, or to multiple docker containers.
    - During training, parameters, metrics and artifacts produced by each training job are tracked using MLflow.
    - Models and artifacts are stored in AWS S3, while metrics are stored in PostgreSQL.
      - MLFlow supports other artifact storage options like Azure Blob Storage, Google Cloud Storage, the local filesystem, SFTP, NFS, HDFS, and Databricks DBFS.
      - MLFlow also supports multiple backends, including PostgreSQL, MySQL, and SQLite.

2. **Deployment Phase:**
    - The Model Selector chooses the best model based on some chosen metrics and parameters for each model and retrieves its runid that is used for retrieval and deployment.
    - The model deployment orchestrator deploys the best model to a Docker container or to a Kubernetes cluster where it creates multiple replicas connected to an Ingress.
    - Inference requests from users are routed through a REST API and Ingress to the deployed model pods.

## Setup & Usage

### 1. Start Infrastructure

```bash
make mlflow-init
```

Starts MLFlow, PostgreSQL, MinIO

### 2. Run Training Jobs

Docker-based training:

```bash
make train-docker
```

Kubernetes-based training:

```bash
make train-kubernetes
```

### 3. Model Selection

```bash
make select-model
```

Selects the best model based on filters defined in `src/model_query/metrics_config.yaml`.

### 4. Deploy Models

Docker deployment:

```bash
make prod-docker
```

Kubernetes deployment (with replica scaling):

```bash
make prod-kubernetes
```

### 5. Inference - Single Input

```bash
make inference-one-input
```

## Makefile Targets

| Target            | Description                          |
|-------------------|--------------------------------------|
| mlflow-init       | Start MLFlow infrastructure          |
| mlflow-remove     | Stop and clean MLFlow infrastructure |
| build-train       | Build training container             |
| build-prod        | Build production container           |
| train-docker      | Run training in multiple Docker containers (single node)               |
| train-kubernetes  | Run training in multiple Kubernetes pods (multiple nodes)                |
| select-model      | Select best model                    |
| prod-docker       | Deploy model in Docker container     |
| prod-kubernetes   | Deploy model in Kubernetes           |
| inference-one     | Run inference on single input        |
| generate-data     | Generate synthetic data              |
| compose-logs      | View container logs                  |

## Configuration

- Set model registry location in `src/orchestrators/`
- Configure Kubernetes manifests for scaling
- Update MinIO credentials in `compose.yaml`
