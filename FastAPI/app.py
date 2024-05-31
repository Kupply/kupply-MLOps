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

    if month <= 8:
        return f"{year}_1"
    else:
        return f"{year}_2"

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/prediction", response_model=PredictOutput)
async def inference(item_list: DataInput):
    item_dict = {'firstMajor': item_list.firstMajor, 
                 'applyGrade': item_list.applyGrade, 
                 'applyMajor': item_list.applyMajor, 
                 'applySemester': item_list.applySemester, 
                 'applyGPA': item_list.applyGPA}
    df = pd.DataFrame([item_dict])

    current_semester = get_current_semester()
    classification_model = load_model(model_name=f"{current_semester}_classification_model",
                                      platform='aws', 
                                      authentication={'bucket': AWS_BUCKET_NAME, 
                                                      'path': "models/"})
    prediction = predict_model(classification_model, data=df)

    return {"prediction_label": prediction.iloc[0]['prediction_label'],
            "prediction_score": prediction.iloc[0]['prediction_score']}