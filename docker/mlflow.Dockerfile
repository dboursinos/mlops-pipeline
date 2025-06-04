FROM python:3.13-slim-bullseye

RUN pip install mlflow psycopg2-binary boto3 cryptography
