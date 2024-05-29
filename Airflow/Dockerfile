FROM apache/airflow:2.9.1-python3.11

# Switch to root to install system dependencies
USER root

RUN apt-get update && apt-get install -y libgomp1

# Switch back to the airflow user
USER airflow

COPY requirements.txt /

RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" -r /requirements.txt