from io import StringIO
from datetime import datetime

import pandas as pd
from pycaret.classification import setup, compare_models, finalize_model, save_model, load_model
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

def get_current_semester():
    now = datetime.now()
    year = now.year
    month = now.month

    if month <= 8:
        return f"{year}_1"
    else:
        return f"{year}_2"

def get_data_from_s3():
    current_semester = get_current_semester()

    hook = S3Hook('aws_default')

    key = f'applications/{current_semester}_applications.csv'
    bucket_name = 'kupply-bucket'

    applications_csv = hook.read_key(key=key, bucket_name=bucket_name)
    applications_df = pd.read_csv(StringIO(applications_csv))

    return applications_df

def train_model(ti):
    applications_df = ti.xcom_pull(task_ids='get_data_from_s3_task')

    classification_s = setup(data=applications_df, target='pass', session_id=320107)

    classification_best_model = compare_models()

    print(f"Best Model: {classification_best_model}")

    classification_best_model_final = finalize_model(classification_best_model)

    # Save the model to a file
    model_filename = '/tmp/classification_best_model_final'
    save_model(classification_best_model_final, model_filename)
    
    return model_filename

def upload_model_to_s3(ti):
    model_filename = ti.xcom_pull(task_ids='train_model_task')
    model_path = f'{model_filename}.pkl'

    current_semester = get_current_semester()

    hook = S3Hook('aws_default')
    key = f'models/{current_semester}_classification_model.pkl'
    bucket_name = 'kupply-bucket'
    hook.load_file(filename=model_path, key=key, bucket_name=bucket_name, replace=True)

    return