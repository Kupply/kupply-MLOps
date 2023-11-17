from fastapi import FastAPI
from typing import List
import pandas as pd
from config import Config
from classifier import classifier
from model.item import DataInput, PredictOutput
from preprocessor import cls_inference_preprocess, tokenize, get_dataloader

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict", response_model=PredictOutput)
async def classify_blog_text(item_list: DataInput):
    df = pd.DataFrame(item_list)
    processed_str = cls_inference_preprocess(df)
    inference_tokenized_data = tokenize(processed_str)
    inference_dataloader = get_dataloader(inference_tokenized_data)

    prediction = classifier(inference_dataloader)

    return prediction