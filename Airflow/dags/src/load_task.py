import pandas as pd
from datetime import datetime

from airflow.providers.mongo.hooks.mongo import MongoHook
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

def get_current_semester():
    now = datetime.now()
    year = now.year
    month = now.month

    if month <= 8:
        return f"{year}-1"
    else:
        return f"{year}-2"
    
def get_application_data():
    hook = MongoHook(mongo_conn_id="mongo_conn")
    client = hook.get_conn()
    db = (
        client.test
    )

    users = db.users
    applications = db.applications
    majors = db.majors

    print(f"Connected to MongoDB - {client.server_info()}")

    applications_result = applications.find({
        "candidateId": {"$ne": None},
        "applySemester": {"$ne": "2024-1"}
    }, { "_id": 0, "candidateId": 1, "applyGrade": 1, "applySemester": 1, "applyMajor1": 1, "applyGPA": 1, "pnp": 1 })
    users_result = users.find({}, { "_id": 1, "firstMajor": 1 })
    majors_result = majors.find()

    users_dict = {user["_id"]: user for user in users_result}
    applications_dict = {app["candidateId"]: app for app in applications_result}
    majors_dict = {major["_id"]: major["name"] for major in majors_result}

    merged_data = []
    for user_id, user_data in users_dict.items():
        if user_id in applications_dict:
            application_data = applications_dict[user_id]
            application_data["pnp"] = 1 if application_data["pnp"] == "PASS" else 0 # PASS: 1, FAIL or TBD: 0
            user_data["firstMajor"] = majors_dict.get(user_data["firstMajor"], "Unknown")
            application_data["applyMajor1"] = majors_dict.get(application_data["applyMajor1"], "Unknown")
            merged_data.append({**user_data, **application_data})

    application_df = pd.DataFrame(merged_data)
            
    drop_list = ["_id", "candidateId"]
    application_df.drop(labels=drop_list, axis=1, inplace=True)
    application_df.rename(columns = {"pnp": "pass", "applyMajor1": "applyMajor"}, inplace=True)

    return application_df
    
def upload_to_s3(ti):
    application_df = ti.xcom_pull(task_ids='get_application_data_task')
    current_semester = get_current_semester()
    application_df.to_csv(f'{current_semester}_applications.csv', index=False)

    hook = S3Hook('aws_default')
    filename = f'{current_semester}_applications.csv'
    key = f'applications/{current_semester}_applications.csv'
    bucket_name = 'kupply-bucket'
    hook.load_file(filename=filename, key=key, bucket_name=bucket_name, replace=True)