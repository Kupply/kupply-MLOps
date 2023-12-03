# 기본적인 라이브러리
import pandas as pd
import numpy as np
import math
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from os import path
from datetime import datetime

# for koBERT
from kobert_tokenizer import KoBERTTokenizer
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from tqdm import tqdm
import sentencepiece
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import gluonnlp as nlp
from tqdm.notebook import tqdm

tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
model = BertForSequenceClassification.from_pretrained('skt/kobert-base-v1',num_labels=2)

# Trainig Config
# 딥러닝...이지만... 샘플 데이터수가 작은 관계로 전반적으로 작은 값으로 설정
epochs = 50
warmup_ratio = 0.1
lr = 2e-5
grad_clip = 1.0
train_log_interval = 30 # train 이 100번 이루어질 때마다 logging
# validation_interval = 1000
save_interval = 60 # save point는 1000번의 train

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device) # GPU 에 모델 올리기

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# scheduler
data_len = len(train_dataloader)
num_train_steps = int(data_len / batch_size * epochs)
num_warmup_steps = int(num_train_steps * warmup_ratio)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)

# Train 함수 정의
def model_train(model, optimizer, scheduler, train_dataloader, device):
    model.to(device)  # 모델 학습을 설정된 device (CPU, cuda) 위에서 진행하도록 설정

    model.train() # 모델을 학습 모드로 전환
    loss_list_between_log_interval = []

    for epoch_id in range(epochs):
        for step_index, batch_data in tqdm(enumerate(train_dataloader), f"[TRAIN] Epoch:{epoch_id+1}", total=len(train_dataloader)):
                global_step = len(train_dataloader) * epoch_id + step_index + 1

                # Add a condition to break the loop if we've gone through all data points
                if step_index * batch_size >= len(dataset):
                  continue

                optimizer.zero_grad()
                contents = batch_data['content']
                labels = batch_data['label']
                attention_masks = batch_data['attention_mask']

                # 모델의 input들을 device(GPU)와 호환되는 tensor로 변환
                contents = contents.to(device)
                labels = labels.to(device)
                attention_masks = attention_masks.to(device)

                model_outputs = model(
                    contents, token_type_ids=None, attention_mask=attention_masks, labels=labels
                    )

                loss = model_outputs.loss
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                scheduler.step()

                # for logging
                loss_list_between_log_interval.append(model_outputs.loss.item())

                #if global_step % train_log_interval == 0:

        mean_loss = np.mean(loss_list_between_log_interval)

        # 콘솔 출력
        # logger.info(
        #     f"EP:{epoch_id} global_step:{global_step} "
        #     f"loss:{mean_loss:.4f} perplexity:{math.exp(mean_loss):.4f}"
        # )

        loss_list_between_log_interval.clear()

        # if global_step % validation_interval == 0:
        # dev_loss = _validate(model, val_dataloader, device, logger, global_step)

        # 각 epoch 마다 모델 저장
        state_dict = model.state_dict()
        model_path = os.path.join('/content/drive/MyDrive/Colab Notebooks/kupply-MLOps/checkpoint/train_1', f"kupply_epoch_{epoch_id}.pth")
        # logger.info(f"global_step: {global_step} model saved at {model_path}")
        torch.save(state_dict, model_path)

    return model

# Let's Training
model_v1 = model_train(model, optimizer, scheduler, train_dataloader, device)