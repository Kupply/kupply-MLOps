from FastAPI.kobert_tokenizer.kobert_tokenizer import KoBERTTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from airflow.models import TaskInstance

# Train 용 DataLoader 정의


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
    tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
    return tokenizer

# No dependency


def preprocess_dataset(**kwargs):
    sample = pd.read_csv('../data/train_data.csv', index_col=0)  # 데이터셋 삽입
    preprocessed = []
    for _, row in sample.iterrows():
        text = f"First Major is {row['firstMajor']}, Apply Grade is {row['applyGrade']}, Apply Major is {row['applyMajor']}, Apply Semester is {row['applySemester']}, GPA is {row['applyGPA']}, Pass is {row['pass']}"
        preprocessed_text = "[CLS] " + text + " [SEP]"
        preprocessed.append(preprocessed_text)

    task_instance = kwargs['ti']
    task_instance.xcom_push(key='preprocessed_data', value=preprocessed)
    # return preprocessed


def tokenize_dataset(**kwargs):
    tokenizer = get_tokenizer()

    task_instance = kwargs['ti']
    sample = task_instance.xcom_pull(task_ids='preprocessed_data')

    # Checking if data is available
    if sample is None:
        raise ValueError("No data received from 'preprocessed_data' task")

    tokenized_data = tokenizer.batch_encode_plus(
        sample,
        add_special_tokens=True,
        padding='longest',
        truncation=True,
        max_length=128,  # 기존 256 에서 수정
        return_attention_mask=True,
        return_tensors='pt'
    )
    task_instance.xcom_push(key='tokenized_data', value=tokenized_data)
    # return tokenized_data

# No dependency


def get_labels():
    sample = pd.read_csv('../data/train_data.csv', index_col=0)  # 데이터셋 삽입
    # df = # raw_data (DB 혹은 CSV 파일)
    labels = list(map(int, sample['pass'].tolist()))
    print(labels)
    return labels


def split_dataset(**kwargs):
    # dataset = # raw_data

    task_instance = kwargs['ti']
    sample = task_instance.xcom_pull(task_ids='fin_dataset')

    ratio = 0.8  # 임의 설정
    train_size = int(ratio * len(sample))
    val_size = len(sample) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        sample, [train_size, val_size])

    task_instance.xcom_push(key='split_dataset', value=[
                            train_dataset, val_dataset])
    # return train_dataset, val_dataset


def get_train_dataset(**kwargs):
    labels = get_labels()

    task_instance = kwargs['ti']
    sample = task_instance.xcom_pull(task_ids='tokenized_data')

    # Checking if data is available
    if sample is None:
        raise ValueError("No data received from 'preprocessed_data' task")

    fin_dataset = trainDataset(
        content=sample['input_ids'],
        labels=labels,
        attention_masks=sample['attention_mask'],
    )
    task_instance.xcom_push(key='fin_dataset', value=fin_dataset)
    # return fin_dataset


def get_dataloader(**kwargs):

    task_instance = kwargs['ti']
    sample = task_instance.xcom_pull(task_ids='split_dataset')

    train_dataset = sample[0]
    val_dataset = sample[1]

    batch_size = 8
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    # val 데이터셋은 shuffle 하면 안된다.
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)

    task_instance.xcom_push(key='train_dataloader', value=train_dataloader)
    task_instance.xcom_push(key='val_dataloader', value=val_dataloader)
    # return train_dataloader, val_dataloader
