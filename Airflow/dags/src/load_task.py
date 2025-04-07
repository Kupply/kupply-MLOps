from datetime import datetime
from collections import defaultdict

import pandas as pd

from airflow.providers.mongo.hooks.mongo import MongoHook
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

def get_current_semester():
    now = datetime.now()
    year = now.year
    month = now.month

    if month <= 8:
        return f"{year}_1"
    else:
        return f"{year}_2"
    
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
        "pnp": {"$ne": "TBD"},
    }, { "_id": 0, "candidateId": 1, "applyGrade": 1, "applySemester": 1, "applyMajor1": 1, "applyGPA": 1, "pnp": 1 })
    users_result = users.find({}, { "_id": 1, "firstMajor": 1 })
    majors_result = majors.find()

    users_dict = {user["_id"]: user for user in users_result}
    majors_dict = {major["_id"]: major["name"] for major in majors_result}
    
    applications_dict = defaultdict(list)
    for application in applications_result:
        applications_dict[application["candidateId"]].append(application)

    merged_data = []
    for user_id, application_list in applications_dict.items():
        user = users_dict.get(user_id)
        if not user:
            print(f"User with ID {user_id} not found in users collection.")
            print(f"Application data: {application_list}")
            continue

        first_major = majors_dict.get(user["firstMajor"], "Unknown")

        for application_data in application_list:
            apply_major = majors_dict.get(application_data["applyMajor1"], "Unknown")
            merged_data.append({
                "firstMajor": first_major,
                "applyGrade": application_data["applyGrade"],
                "applySemester": application_data["applySemester"],
                "applyMajor": apply_major,
                "applyGPA": application_data["applyGPA"],
                "pass": 1 if application_data["pnp"] == "PASS" else 0
            })

    application_df = pd.DataFrame(merged_data)
    
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