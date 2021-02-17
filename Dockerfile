FROM python:3.7-slim-buster

RUN mkdir /app
COPY requirements.txt requirements.txt
COPY ml_pipeline.py /app/ml_pipeline.py
COPY tests.py /app/tests.py

RUN apt-get update && apt-get install -qq -y postgresql-client curl netcat build-essential libpq-dev --no-install-recommends && \
    pip install --upgrade pip && \
    pip install -r requirements.txt && \
    rm -rf /var/lib/apt/lists/*



