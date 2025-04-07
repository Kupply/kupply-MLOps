from datetime import datetime, timedelta

import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator

from src.train_task import get_data_from_s3, train_model, upload_model_to_s3

local_tz = pendulum.timezone("Asia/Seoul")

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 4, 1, tzinfo=local_tz),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(minutes=10),
}

with DAG(
    dag_id='train_model_dag',
    default_args=default_args,
    description='Train Model with Pycaret and Upload to S3',
    schedule_interval=None,  # get_data_from_mongo에 의해 trigger된다.
    tags=['kupply', 'train_model', 'classification'],
    catchup=False,
) as dag:
    
    get_data_from_s3_task = PythonOperator(
        task_id='get_data_from_s3_task',
        python_callable=get_data_from_s3,
    )

    train_model_task = PythonOperator(
        task_id='train_model_task',
        python_callable=train_model,
    )

    upload_model_to_s3_task = PythonOperator(
        task_id='upload_model_to_s3_task',
        python_callable=upload_model_to_s3,
    )

    get_data_from_s3_task >> train_model_task >> upload_model_to_s3_task