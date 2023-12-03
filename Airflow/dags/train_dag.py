# 1. Train Automation System 구축 

import os
import pendulum # Python datetime module 조작 목적
from datetime import timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# Airflow 폴더 내에서 실행 필요, 오류 시 경로 수정
from src.preprocess_algo import inferenceDataset # 클래스 import 
from src.preprocess_algo import preprocess_raw_data, tokenize_processed_data, get_dataloader # 함수 import 
from src.train_algo import 

seoul_time = pendulum.timezone('Asia/Seoul')
dag_name = os.path.basename(__file__).split('.')[0]
# 'train_dag' 으로 저장

default_args = {
    'owner': 'kupply',
    'retries': 3,
    'retry_delay': timedelta(minutes=1) 
    # 실패 시 1분 간격으로 retry (추후 수정 필요)
}

# 단순 복붙 상태, train_algo.py 작성 이후 수정 필요
with DAG(
    dag_id=dag_name,
    default_args=default_args,
    description='2023-2 DevKor MLOps 프로젝트',
    schedule_interval=timedelta(minutes=10), # 수정 필요
    start_date=pendulum.datetime(2023, 12, 1, tz=seoul_time), # 수정 필요
    catchup=False, # 과거(= 과거 start_date)의 dag 실행 여부
    tags=['kupply', 'classification'] # 필요 시 추가
) as dag:
    get_market_fundamental_task = PythonOperator(
        task_id='get_market_fundamental_task',
        python_callable=get_market_fundamental,
    )
    
    select_columns_task = PythonOperator(
        task_id='select_columns_task',
        python_callable=select_columns,
    )
    
    remove_row_fundamental_task = PythonOperator(
        task_id='remove_row_fundamental_task',
        python_callable=remove_row_fundamental,
    )
    
    rank_fundamental_task = PythonOperator(
        task_id='rank_fundamental_task',
        python_callable=rank_fundamental,
    )
    
    select_stock_task = PythonOperator(
        task_id='select_stock_task',
        python_callable=select_stock,
    )
    
    print_selected_stock_task = PythonOperator(
        task_id='print_selected_stock_task',
        python_callable=print_selected_stock,
    )
    
    # 실행 순서대로 나열
    get_market_fundamental_task >> select_columns_task >> remove_row_fundamental_task >> rank_fundamental_task >> select_stock_task >> print_selected_stock_task