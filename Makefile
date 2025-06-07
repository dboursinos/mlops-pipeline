# Define the Docker Compose file name
DOCKER_COMPOSE_FILE := compose.yaml

# Target to run Docker Compose for the mlflow, postgres and minio services
mlflow-init:
	docker compose -f ./docker/$(DOCKER_COMPOSE_FILE) up -d

# Target to stop Docker Compose and remove mlflow, postgres and minio services
mlflow-remove:
	docker compose -f ./docker/$(DOCKER_COMPOSE_FILE) down --rmi 'all'

compose-logs:
	docker compose -f ./docker/$(DOCKER_COMPOSE_FILE) logs

build-train:
	docker build . -f ./docker/train.Dockerfile -t ml_training
	docker tag ml_training 192.168.1.67:5050/ml-training
	docker push 192.168.1.67:5050/ml-training

build-prod:
	docker build . -f ./docker/prod.Dockerfile -t ml_production
	docker tag ml_production 192.168.1.67:5050/ml-production
	docker push 192.168.1.67:5050/ml-production

train-kubernetes: build-train
	uv run python ./src/orchestrators/kubernetes_orchestrator.py

train-docker: build-train
	uv run python ./src/orchestrators/docker_orchestrator.py

prod-kubernetes: build-prod
	uv run python ./src/orchestrators/kubernetes_model_deployment.py

prod-docker: build-prod
	uv run python ./src/orchestrators/docker_model_deployment.py

inference-one-input:
	uv run python ./src/inference_api/inference_test.py

select-model:
	uv run python ./src/model_query/model_tracker.py

generate-data:
	uv run python ./src/data_gen/data_gen.py
