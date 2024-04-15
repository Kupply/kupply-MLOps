import os
from dotenv import load_dotenv
import boto3

class Config():
    def __init__(self, aws_bucket_name, aws_key):
        self.aws_bucket_name = aws_bucket_name
        self.aws_key = aws_key
        self.model_path = "./trained_model/downloaded_model.pth" 
        self._download_model()

    def _download_model(self):
        if not os.path.exists(self.model_path):
            client = boto3.client('s3',
                                  aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                                  aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                                  region_name=os.getenv('AWS_DEFAULT_REGION')
                                  )
            client.download_file(self.aws_bucket_name, self.aws_key, self.model_path)

    def get_model_path(self):
        return self.model_path

# 서버 시작 시, 모델이 없을 때 다운로드
load_dotenv()
aws_bucket_name = os.getenv('AWS_BUCKET_NAME')
aws_key = 'models/kupply_epoch_49.pth'
config = Config(aws_bucket_name, aws_key)
model_path = config.get_model_path()