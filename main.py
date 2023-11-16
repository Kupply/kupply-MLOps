import argparse
import torch

from fastapi import FastAPI
from packages import FastAPIRunner

from packages.config import DataInput, PredictOutput
from packages.config import ProjectConfig

app = FastAPI()

# Project config 설정
project_config = ProjectConfig('ncf')
# 모델 가져오기
model = project_config.load_model()
model.eval()

@app.get('/')
def read_results():
    return {'msg' : 'Main'}

@app.post('/predict', response_model=PredictOutput)
async def predict(data_request: DataInput):
    user_id = data_request.user_id
    item_id = data_request.movie_id
    predict = model(torch.tensor( [[user_id, item_id]] ))
    prob, prediction = predict, int(( predict > project_config.threshold ).float() * 1) 
    return {'prob' : prob, 'prediction' : prediction}
    
if __name__ == "__main__":
    # python main.py --host 127.0.0.1 --port 8000
    parser = argparse.ArgumentParser()
    parser.add_argument('--host')
    parser.add_argument('--port')
    args = parser.parse_args()
    api = FastAPIRunner(args)
    api.run()