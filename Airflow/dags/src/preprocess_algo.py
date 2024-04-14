# from kobert_tokenizer.kobert_tokenizer import KoBERTTokenizer
import torch
import pandas as pd
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from io import StringIO

# Airflow tokenizer folder 인식 불가 이슈 해결 목적 임시 라이브러리 임포트
from transformers import AutoTokenizer

class trainDataset(Dataset):
    def __init__(self, content, labels, attention_masks):
        self.content = content
        self.labels = labels
        self.attention_masks = attention_masks
        # 훈련 데이터셋에 라벨이 2 class 인 관계로, 우선 2 class 로 설정 (코드 수정 시, 위 토크나이저 임포트 코드와 함께 바꿔주기)
        self.num_classes = 2
        self.one_hot_labels = torch.zeros(len(labels), self.num_classes)
        for i, label in enumerate(self.labels):
            # print(i,label)
            self.one_hot_labels[i, label] = 1

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):
        return {
            'content': self.content[idx],
            'label': self.one_hot_labels[idx],
            'attention_mask': self.attention_masks[idx],
            'gt_label': self.labels[idx]
        }

def get_tokenizer():
    # tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    return tokenizer

def get_data_from_s3():
    date = datetime.now().strftime("%Y%m%d")
    hook = S3Hook('aws_default')
    key = f'applications/{date}_applications.csv'
    bucket_name = 'kupply-bucket'
    res = hook.read_key(key=key, bucket_name=bucket_name)
    return res

def get_labels(ti):
    csv_sample = ti.xcom_pull(task_ids='get_data_from_s3_task')
    sample = pd.read_csv(StringIO(csv_sample))
    labels = list(map(int, sample['pass'].tolist()))

    return labels

def preprocess_dataset(ti):
    csv_sample = ti.xcom_pull(task_ids='get_data_from_s3_task')
    sample = pd.read_csv(StringIO(csv_sample))

    preprocessed = []
    for _, row in sample.iterrows():
        text = f"First Major is {row['firstMajor']}, Apply Grade is {row['applyGrade']}, Apply Major is {row['applyMajor']}, Apply Semester is {row['applySemester']}, GPA is {row['applyGPA']}, Pass is {row['pass']}"
        preprocessed_text = "[CLS] " + text + " [SEP]"
        preprocessed.append(preprocessed_text)
    
    return preprocessed


def tokenize_dataset(ti):
    tokenizer = get_tokenizer()
    sample = ti.xcom_pull(task_ids='preprocess_dataset_task')

    # Checking if data is available
    if sample is None:
        raise ValueError("No data received from 'preprocess_dataset_task' task")

    tokenized_data = tokenizer.batch_encode_plus(
        sample,
        add_special_tokens=True,
        padding='longest',
        truncation=True,
        max_length=128,  # 기존 256 에서 수정
        return_attention_mask=True,
        return_tensors='pt'
    )

     # Extracting necessary information for serialization to JSON
    serialized_data = {
        'input_ids': tokenized_data['input_ids'].tolist(),
        'attention_mask': tokenized_data['attention_mask'].tolist(),
    }

    return serialized_data

def get_train_dataset(ti):
    labels = ti.xcom_pull(task_ids='get_labels_task')
    sample = ti.xcom_pull(task_ids='tokenize_dataset_task')

    # Checking if data is available
    if sample is None or labels is None:
        raise ValueError("No data received from 'tokenize_dataset_task' task")

    # Initialize TrainDataset with the serialized data
    fin_dataset = trainDataset(
        content=sample['input_ids'],
        labels=labels,
        attention_masks=sample['attention_mask'],
    )

    return fin_dataset

def split_dataset(ti):
    sample = ti.xcom_pull(task_ids='get_train_dataset_task')

    ratio = 0.8  # 임의 설정
    train_size = int(ratio * len(sample))
    val_size = len(sample) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        sample, [train_size, val_size])

    return train_dataset, val_dataset

def get_dataloader(ti):
    sample = ti.xcom_pull(task_ids='split_dataset_task')

    train_dataset = sample[0]
    val_dataset = sample[1]

    batch_size = 8
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    # val 데이터셋은 shuffle 하면 안된다.
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, val_dataloader
