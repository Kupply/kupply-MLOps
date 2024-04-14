import os
from datetime import datetime, timedelta
import pendulum  # Python datetime module 조작 목적

from airflow import DAG
from airflow.operators.python import PythonOperator

from src.preprocess_algo import get_tokenizer, get_labels, preprocess_dataset, tokenize_dataset, get_data_from_s3, split_dataset, get_train_dataset, get_dataloader  # 함수 import
from src.train_algo import get_model_config, get_model, get_train_config, get_optimizer, get_scheduler_config, get_scheduler, model_train


local_tz = pendulum.timezone("Asia/Seoul")
dag_name = os.path.basename(__file__).split('.')[0]
# 'train_dag' 으로 저장

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 4, 1, tzinfo=local_tz),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(minutes=10),
}

# 단순 복붙 상태
with DAG(
    dag_id=dag_name,
    default_args=default_args,
    description='2023-2 DevKor MLOps 프로젝트',
    schedule_interval=None,  # get_data_from_mongo에 의해 trigger된다.
    catchup=False,
    tags=['kupply', 'classification']  # 필요 시 추가
) as dag:
    
    get_data_from_s3_task = PythonOperator(
        task_id='get_data_from_s3_task',
        provide_context=True,
        python_callable=get_data_from_s3,  # 함수에 인자(sample) 필요
    )

    get_labels_task = PythonOperator(
        task_id='get_labels_task',
        provide_context=True,
        python_callable=get_labels,  # 함수에 인자(sample) 필요
    )

    preprocess_dataset_task = PythonOperator(
        task_id='preprocess_dataset_task',
        provide_context=True,
        python_callable=preprocess_dataset,  # 함수에 인자(sample) 필요
    )

    tokenize_dataset_task = PythonOperator(
        task_id='tokenize_dataset_task',
        provide_context=True,
        python_callable=tokenize_dataset,  # 함수에 인자(sample) 필요
    )

    get_train_dataset_task = PythonOperator(
        task_id='get_train_dataset_task',
        provide_context=True,
        python_callable=get_train_dataset,  # 함수에 인자(sample) 필요
    )

    split_dataset_task = PythonOperator(
        task_id='split_dataset_task',
        provide_context=True,
        python_callable=split_dataset,  # 함수에 인자(sample, ratio) 2개 필요
    )

    get_dataloader_task = PythonOperator(
        task_id='get_dataloader_task',
        provide_context=True,
        python_callable=get_dataloader,  # 함수에 인자(sample, ratio) 2개 필요
    )

    get_train_config_task = PythonOperator(
        task_id='get_train_config_task',
        provide_context=True,
        python_callable=get_train_config,
    )

    get_optimizer_task = PythonOperator(
        task_id='get_optimizer_task',
        provide_context=True,
        python_callable=get_optimizer,
    )

    get_scheduler_config_task = PythonOperator(
        task_id='get_scheduler_config_task',
        provide_context=True,
        python_callable=get_scheduler_config,  # 함수에 인자(train_dataloader) 필요
    )

    get_scheduler_task = PythonOperator(
        task_id='get_scheduler_task',
        provide_context=True,
        python_callable=get_scheduler,
    )

    model_train_task = PythonOperator(
        task_id='model_train_task',
        provide_context=True,
        python_callable=model_train,  # 함수에 인자(train_dataloader) 필요
    )

    get_data_from_s3_task >> get_labels_task >> preprocess_dataset_task >> tokenize_dataset_task >> get_train_dataset_task >> split_dataset_task >> get_dataloader_task 
    get_dataloader_task >> get_train_config_task >> get_optimizer_task >> get_scheduler_config_task >> get_scheduler_task >> model_train_task

    """
    [ XCom 설명 ]
    python task의 경우, python_callable로 호출한 함수에서 return하는 값이 있을 경우 자동으로 xcom_push() 가 실행됨. 즉 return 값이 자동으로 xcom에 저장됨.
    다음 task에서 이전 task의 return 값을 사용하려면 xcom_pull()을 사용함.
    provide_context=True로 설정해야 함.
    
    """
