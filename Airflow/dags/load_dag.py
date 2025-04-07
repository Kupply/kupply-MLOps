import pendulum
from datetime import datetime, timedelta

import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

from src.load_task import get_application_data, upload_to_s3

local_tz = pendulum.timezone("Asia/Seoul")

default_args = {
    'owner': 'kupply',
    'depends_on_past': False,
    'start_date': datetime(2023, 4, 1, tzinfo=local_tz),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(seconds=60),
}

with DAG(
    dag_id='mongo_to_s3_dag',
    default_args=default_args,
    description='Read data from MongoDB and upload to S3',
    schedule_interval="0 0 1 3,9 *", # 매년 3월 1일, 9월 1일 새벽 12시에 실행 => 새 학기마다
    tags=['kupply', 'load_data'],
    catchup=False,
) as dag:
    
    get_application_data_task = PythonOperator(
        task_id='get_application_data_task',
        python_callable=get_application_data,
    )
    
    upload_to_s3_task = PythonOperator(
        task_id='upload_to_s3_task',
        python_callable=upload_to_s3,
    )

    trigger_train_model_dag_task = TriggerDagRunOperator(
        task_id='trigger_train_model_dag_task',
        trigger_dag_id='train_model_dag',
        reset_dag_run=False,
        wait_for_completion=False,
    )

    get_application_data_task >> upload_to_s3_task >> trigger_train_model_dag_task