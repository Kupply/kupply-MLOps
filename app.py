from fastapi import FastAPI
from typing import List
from config import Config
from classifier import classify
from model.item import DataInput, PredictOutput
from preprocessor import preprocess

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict", response_model=PredictOutput)
async def classify_blog_text(item_list: DataInput):
    """
        네이버 블로그 맛집 리뷰 텍스트의 광고 여부를 predict
    """
    item_list = sorted(item_list, key=lambda item: item.id)
    lines_for_predict = []
    for item in item_list:
        # lines_for_predict.append(preprocess(item.fullText))
        # input format 제한 걸어놓았으니 preprocessing 과정은 불필요?
        lines_for_predict.append(item.fullText) 
    config = Config(model_fn="./trained_model/bert_clean.tok.slice.pth", gpu_id=-1, batch_size=8,
                    lines=lines_for_predict)
    classified_lines = classify(config)
    for i, classified_line in enumerate(classified_lines):
        item_list[i].probability = classified_line[0]
        item_list[i].ad = classified_line[1]
    return item_list
    print(item_list)

    if(item_list.applyGPA >= 4.0): 
        return {'result': 1} 
    else:
        return {'result': 0}