import pandas as pd
import numpy as np
import torch
import math
import os

from os import path
from transformers import BertForSequenceClassification

from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from tqdm import tqdm
from tqdm.notebook import tqdm
from torch import nn
import torch.nn.functional as F
import torch.optim as optim


def get_model_config():
    model_path = 'skt/kobert-base-v1'
    num_labels = 2
    return model_path, num_labels


def get_model():
    model_path, num_labels = get_model_config()
    model = BertForSequenceClassification.from_pretrained(
        model_path, num_labels=num_labels)
    return model


def get_train_config():
    epochs = 50
    lr = 2e-5
    grad_clip = 1.0
    train_log_interval = 30
    # validation_interval = 1000
    save_interval = 60
    batch_size = 8
    return epochs, lr, grad_clip, train_log_interval, save_interval, batch_size  # 총 6가지


def get_optimizer(ti):
    model = get_model()
    _, lr, _, _, _, batch_size = ti.xcom_pull(task_ids='get_train_config_task')
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    return optimizer


def get_scheduler_config(ti):  # 보호관찰 필요=
    dataloader = ti.xcom_pull(task_ids='get_dataloader_task')
    train_dataloader = dataloader[0]

    epochs, _, _, _, _, batch_size = ti.xcom_pull(task_ids='get_train_config_task')
    warmup_ratio = 0.1
    data_len = len(train_dataloader)
    num_train_steps = int(data_len / batch_size * epochs)
    num_warmup_steps = int(num_train_steps * warmup_ratio)

    return num_train_steps, num_warmup_steps


def get_scheduler(ti):
    num_train_steps, num_warmup_steps = ti.xcom_pull(task_ids='get_scheduler_config_task')
    optimizer = ti.xcom_pull(task_ids='get_optimizer_task')
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
    return scheduler

# Train 함수 정의


def model_train(ti):
    dataloader = ti.xcom_pull(task_ids='get_dataloader_task')
    train_dataloader = dataloader[0]

    # model config
    model = get_model()
    optimizer = ti.xcom_pull(task_ids='get_optimizer_task')
    scheduler = ti.xcom_pull(task_ids='get_scheduler_task')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # train config
    epochs, lr, grad_clip, train_log_interval, save_interval, batch_size = ti.xcom_pull(task_ids='get_train_config_task')

    # 모델 학습을 설정된 device (CPU, cuda) 위에서 진행하도록 설정
    model.to(device)

    # 모델을 학습 모드로 전환
    model.train()
    loss_list_between_log_interval = []

    # 모델 이터레이션
    for epoch_id in range(epochs):
        for step_index, batch_data in tqdm(enumerate(train_dataloader), f"[TRAIN] Epoch:{epoch_id+1}", total=len(train_dataloader)):
            global_step = len(train_dataloader) * epoch_id + step_index + 1

            # Add a condition to break the loop if we've gone through all data points
            if step_index >= len(train_dataloader):  # len(dataset):
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

            # if global_step % train_log_interval == 0:

        mean_loss = np.mean(loss_list_between_log_interval)

        # 콘솔 출력
        print(f"EP:{epoch_id} global_step:{global_step}")
        print(f"loss:{mean_loss:.4f}")
        # perplexity:{math.exp(mean_loss):.4f}")

        # logger.info(
        #     f"EP:{epoch_id} global_step:{global_step} "
        #     f"loss:{mean_loss:.4f} perplexity:{math.exp(mean_loss):.4f}"
        # )

        loss_list_between_log_interval.clear()

        # if global_step % validation_interval == 0:
        # dev_loss = _validate(model, val_dataloader, device, logger, global_step)

        # 각 epoch 마다 모델 저장
        state_dict = model.state_dict()
        model_path = os.path.join(
            '../checkpoints/', f"kupply_epoch_{epoch_id}.pth")
        # logger.info(f"global_step: {global_step} model saved at {model_path}")
        torch.save(state_dict, model_path)

    # return model


# Let's Training
# model_v1 = model_train(model, optimizer, scheduler, train_dataloader, device)
