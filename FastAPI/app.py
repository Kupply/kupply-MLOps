import os
from datetime import datetime

import pandas as pd
from fastapi import FastAPI
from dotenv import load_dotenv
from pycaret.classification import load_model, predict_model

from model.item import DataInput, PredictOutput

app = FastAPI(swagger_ui_parameters={"displayRequestDuration": True})

load_dotenv()
AWS_BUCKET_NAME=os.getenv('AWS_BUCKET_NAME')

def get_current_semester():
    now = datetime.now()
    year = now.year
    month = now.month
    return f"{year}_1" if month <= 8 else f"{year}_2"

@app.on_event("startup")
def load_model_on_startup():
    current_semester = get_current_semester()
    model = load_model(
        model_name=f"{current_semester}_classification_model",
        platform='aws',
        authentication={
            'bucket': AWS_BUCKET_NAME,
            'path': "models/"
        }
    )
    app.state.model = model

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/prediction", response_model=PredictOutput)
def inference(item_list: DataInput):
    item_dict = {
        'firstMajor': item_list.firstMajor,
        'applyGrade': item_list.applyGrade,
        'applyMajor': item_list.applyMajor,
        'applySemester': item_list.applySemester,
        'applyGPA': item_list.applyGPA
    }
    df = pd.DataFrame([item_dict])

    model = app.state.model
    prediction = predict_model(model, data=df)

    return {
        "prediction_label": prediction.iloc[0]['prediction_label'],
        "prediction_score": prediction.iloc[0]['prediction_score']
    }