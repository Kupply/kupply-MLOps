import os
from dotenv import load_dotenv
from fastapi import FastAPI
from typing import List
import pandas as pd
from config import Config
from classifier import classifier
from model.item import DataInput, PredictOutput
from preprocessor import cls_inference_preprocess, tokenize, get_dataloader

app = FastAPI()

load_dotenv()
AWS_BUCKET_NAME=os.getenv('AWS_BUCKET_NAME')

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict", response_model=PredictOutput)
async def inference(item_list: DataInput):
    item_dict = {'firstMajor': item_list.firstMajor, 'applyGrade': item_list.applyGrade, 'applyMajor': item_list.applyMajor, 'applySemester': item_list.applySemester, 'applyGPA': item_list.applyGPA}
    df = pd.DataFrame([item_dict])
    processed_str = cls_inference_preprocess(df)
    inference_tokenized_data = tokenize(processed_str)
    inference_dataloader = get_dataloader(inference_tokenized_data)

    prediction = classifier(Config(aws_bucket_name=AWS_BUCKET_NAME, aws_key='models/kupply_epoch_49.pth'), inference_dataloader)

    return {"result": prediction}