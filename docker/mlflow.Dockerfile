FROM python:3.13-slim-bullseye

RUN pip install mlflow==2.22.1 psycopg2-binary==2.9.10 boto3==1.38.38 cryptography==45.0.4
