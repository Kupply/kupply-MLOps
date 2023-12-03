from kobert_tokenizer import KoBERTTokenizer # 별도 처리 필요
import torch
from torch.utils.data import Dataset, DataLoader

# tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
# Train 용 DataLoader 정의 (train 파일에서 함께 import)
class trainDataset(Dataset):
    def __init__(self, content, labels, attention_masks):
        self.content = content
        self.labels = labels
        self.attention_masks = attention_masks
        self.num_classes = 2 # 훈련 데이터셋에 라벨이 2 class 인 관계로, 우선 2 class 로 설정 (코드 수정 시, 위 토크나이저 임포트 코드와 함께 바꿔주기)
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
        

# Train 용 전처리 함수
def preprocess_dataset(sample):
    preprocessed = []
    
    for _, row in sample.iterrows():
        text = f"First Major is {row['firstMajor']}, Apply Grade is {row['applyGrade']}, Apply Major is {row['applyMajor']}, Apply Semester is {row['applySemester']}, GPA is {row['applyGPA']}, Pass is {row['pass']}"
        preprocessed_text = "[CLS] " + text + " [SEP]"
        preprocessed.append(preprocessed_text)
        
    return preprocessed


# 토크나이저 함수
def tokenize_dataset(processed_str):
    tokenized_data = tokenizer.batch_encode_plus(
        processed_str, 
        add_special_tokens=True,
        padding='longest',
        truncation=True,
        max_length=128, # 기존 256 에서 수정
        return_attention_mask=True,
        return_tensors='pt'
    )
    return tokenized_data


def get_labels():
    df = # raw_data (DB 혹은 CSV 파일)
    labels = list(map(int,df['pass'].tolist()))
    print(labels)
    
    return labels


def split_dataset(dataset, ratio):
    # dataset = # raw_data
    # ratio = # ratio
    train_size = int(ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    return train_dataset, val_dataset


def get_train_dataset(): 
    raw_data = # raw_data (DB 혹은 CSV 파일)
    tokenized_data = # 추후 위 함수와 연결
    labels = get_labels()
    training_dataset = trainDataset(
        content=tokenized_data['input_ids'],
        labels=labels,
        attention_masks=tokenized_data['attention_mask'],
    )
    return training_dataset
    
    
def get_dataloader():
    dataset = # raw_data
    train_dataset, val_dataset = split_dataset(dataset, 0.8)
    batch_size = 8
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) # val 데이터셋은 shuffle 하면 안된다.

    return train_dataloader, val_dataloader
