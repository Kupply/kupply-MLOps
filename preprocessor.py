from kobert_tokenizer import KoBERTTokenizer
from torch.utils.data import Dataset, DataLoader

tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')

# Inference용 DataLoader 정의
class inferenceDataset(Dataset):
    def __init__(self, content, attention_masks):
        self.content = content
        self.attention_masks = attention_masks
        self.num_classes = 2 # 훈련 데이터셋에 라벨이 2 class 인 관계로, 우선 2 class 로 설정 (코드 수정 시, 위 토크나이저 임포트 코드와 함께 바꿔주기)

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):
        return {
            'content': self.content[idx],
            'attention_mask': self.attention_masks[idx]
        }

# Inference 용 전처리 함수
def cls_inference_preprocess(sample):
    text = f"First Major is {sample['firstMajor']}, Apply Grade is {sample['applyGrade']}, Apply Major is {sample['applyMajor']}, Apply Semester is {sample['applySemester']}, GPA is {sample['applyGPA']}"

    preprocessed = "[CLS] " + text + " [SEP]"

    return preprocessed

# 토크나이징
def tokenize(processed_str):
    inference_tokenized_data = tokenizer.batch_encode_plus(
        processed_str, # lyrics (수정)
        add_special_tokens=True,
        padding='longest',
        truncation=True,
        max_length=256, # 수정
        return_attention_mask=True,
        return_tensors='pt'
    )

    return inference_tokenized_data

def get_dataloader(inference_tokenized_data):
    inference_dataset = inferenceDataset(
        content=inference_tokenized_data['input_ids'],
        attention_masks=inference_tokenized_data['attention_mask'],
    )

    batch_size = 1
    inference_dataloader = DataLoader(inference_dataset, batch_size= batch_size, shuffle=False)

    return inference_dataloader
