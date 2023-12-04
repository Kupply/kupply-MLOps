import os
import pendulum  # Python datetime module 조작 목적
from datetime import timedelta

###
import pandas as pd
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from os import path
from datetime import datetime

from kobert_tokenizer import KoBERTTokenizer
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from tqdm import tqdm
import sentencepiece

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm.notebook import tqdm

###

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator, ShortCircuitOperator


from src.preprocess_algo import trainDataset  # 클래스 import
from src.preprocess_algo import get_tokenizer, preprocess_dataset, tokenize_dataset, get_labels, split_dataset, get_train_dataset, get_dataloader  # 함수 import
from src.train_algo import get_model_config, get_model, get_train_config, get_optimizer, get_scheduler_config, get_scheduler, model_train


#######################################################


seoul_time = pendulum.timezone('Asia/Seoul')
dag_name = os.path.basename(__file__).split('.')[0]
# 'train_dag' 으로 저장

default_args = {
    'owner': 'kupply',
    'retries': 3,
    'retry_delay': timedelta(minutes=1)
    # 실패 시 1분 간격으로 retry (추후 수정 필요)
}

# 단순 복붙 상태
with DAG(
    dag_id=dag_name,
    default_args=default_args,
    description='2023-2 DevKor MLOps 프로젝트',
    schedule_interval=timedelta(minutes=10),  # 수정 필요
    start_date=pendulum.datetime(2023, 12, 1, tz=seoul_time),  # 수정 필요
    catchup=False,  # 과거(= 과거 start_date)의 dag 실행 여부
    tags=['kupply', 'classification']  # 필요 시 추가
) as dag:

    get_model_task = PythonOperator(
        task_id='get_model_task',
        python_callable=get_model,
    )

    get_tokenizer_task = PythonOperator(
        task_id='get_tokenizer_task',
        python_callable=get_tokenizer,
    )

    get_labels_task = PythonOperator(
        task_id='get_labels_task',
        python_callable=get_labels,  # 함수에 인자(sample) 필요
    )

    preprocess_dataset_task = PythonOperator(
        task_id='preprocess_dataset_task',
        python_callable=preprocess_dataset,  # 함수에 인자(sample) 필요
    )

    tokenize_dataset_task = PythonOperator(
        task_id='tokenize_dataset_task',
        python_callable=tokenize_dataset,  # 함수에 인자(sample) 필요
    )

    get_train_dataset_task = PythonOperator(
        task_id='get_train_dataset_task',
        python_callable=get_train_dataset,  # 함수에 인자(sample) 필요
    )

    split_dataset_task = PythonOperator(
        task_id='split_dataset_task',
        python_callable=split_dataset,  # 함수에 인자(sample, ratio) 2개 필요
    )

    get_train_config_task = PythonOperator(
        task_id='get_train_config_task',
        python_callable=get_train_config,
    )

    get_optimizer_task = PythonOperator(
        task_id='get_optimizer_task',
        python_callable=get_optimizer,
    )

    get_scheduler_config_task = PythonOperator(
        task_id='get_scheduler_config_task',
        python_callable=get_scheduler_config,  # 함수에 인자(train_dataloader) 필요
    )

    get_scheduler_task = PythonOperator(
        task_id='get_scheduler_task',
        python_callable=get_scheduler,
    )

    model_train_task = PythonOperator(
        task_id='model_train_task',
        python_callable=model_train,  # 함수에 인자(train_dataloader) 필요
    )

    # 실행 순서대로 나열
    get_model_task >> get_tokenizer_task >> get_labels_task >> preprocess_dataset_task >> tokenize_dataset_task >> get_train_dataset_task >> split_dataset_task >> get_train_config_task >> get_optimizer_task >> get_scheduler_config_task >> get_scheduler_task >> model_train_task
