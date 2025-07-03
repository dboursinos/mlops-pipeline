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

build-train-random-forest:
	docker build . -f ./docker/random_forest.Dockerfile -t 192.168.1.67:5050/ml_training_random_forest:latest
	docker push 192.168.1.67:5050/ml_training_random_forest:latest

build-train-svm:
	docker build . -f ./docker/svm.Dockerfile -t 192.168.1.67:5050/ml_training_svm:latest
	docker push 192.168.1.67:5050/ml_training_svm:latest

build-train-logistic-regression:
	docker build . -f ./docker/logistic_regression.Dockerfile -t 192.168.1.67:5050/ml_training_logistic_regression:latest
	docker push 192.168.1.67:5050/ml_training_logistic_regression:latest

build-train-xgboost:
	docker build . -f ./docker/xgboost.Dockerfile -t 192.168.1.67:5050/ml_training_xgboost:latest
	docker push 192.168.1.67:5050/ml_training_xgboost:latest

build-train-pytorch:
	docker build . -f ./docker/pytorch.Dockerfile -t 192.168.1.67:5050/ml_training_pytorch_mlp:latest
	docker push 192.168.1.67:5050/ml_training_pytorch_mlp:latest

build-train-all: build-train-random-forest build-train-svm build-train-logistic-regression build-train-xgboost build-train-pytorch

build-prod:
	docker build . -f ./docker/prod.Dockerfile -t ml_production
	docker tag ml_production 192.168.1.67:5050/ml-production
	docker push 192.168.1.67:5050/ml-production

train-kubernetes:
	@uv run python ./src/orchestrators/kubernetes_orchestrator.py

train-docker: build-train
	@uv run python ./src/orchestrators/docker_orchestrator.py

prod-kubernetes: build-prod
	@uv run python ./src/orchestrators/kubernetes_model_deployment.py

prod-docker: build-prod
	@uv run python ./src/orchestrators/docker_model_deployment.py

inference-one-input:
	@uv run python ./src/inference_api/inference_test.py

select-model:
	@uv run python ./src/model_query/model_selector.py

generate-data:
	@uv run python ./src/data_gen/data_gen.py
