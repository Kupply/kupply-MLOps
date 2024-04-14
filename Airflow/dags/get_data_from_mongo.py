import pendulum
import pandas as pd
from airflow import DAG
from airflow.decorators import task
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.providers.mongo.hooks.mongo import MongoHook
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from datetime import datetime, timedelta

local_tz = pendulum.timezone("Asia/Seoul")

default_args = {
    'owner': 'kupply',
    'depends_on_past': False,
    'start_date': datetime(2023, 4, 1, tzinfo=local_tz),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(seconds=60),
}

def upload_to_s3(ti):
    application_df = ti.xcom_pull(task_ids='get_application_data')
    date = datetime.now().strftime("%Y%m%d")
    application_df.to_csv(f'{date}_applications.csv', index=False)

    hook = S3Hook('aws_default')
    filename = f'{date}_applications.csv'
    key = f'applications/{date}_applications.csv'
    bucket_name = 'kupply-bucket'
    hook.load_file(filename=filename, key=key, bucket_name=bucket_name, replace=True)


with DAG(
    dag_id='mongo_to_s3_dag',
    default_args=default_args,
    description='Read data from MongoDB and upload to S3',
    schedule_interval="0 0 1 3,9 *", # 매년 3월 1일, 9월 1일 새벽 12시에 실행 => 새 학기마다
    tags=['kupply', 'load_data'],
    catchup=False,
) as dag:

    @task()
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
            "$or": [{"pnp": "PASS"}, {"pnp": "FAIL"}],
            "candidateId": {"$ne": None}
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
                application_data["pnp"] = 1 if application_data["pnp"] == "PASS" else 0
                user_data["firstMajor"] = majors_dict.get(user_data["firstMajor"], "Unknown")
                application_data["applyMajor1"] = majors_dict.get(application_data["applyMajor1"], "Unknown")
                merged_data.append({**user_data, **application_data})

        application_df = pd.DataFrame(merged_data)
                
        drop_list = ["_id", "candidateId"]
        application_df.drop(labels=drop_list, axis=1, inplace=True)
        application_df.rename(columns = {"pnp": "pass", "applyMajor1": "applyMajor"}, inplace=True)

        return application_df

    upload_to_s3 = PythonOperator(
        task_id='upload_to_s3',
        python_callable=upload_to_s3,
    )

    trigger_train_dag = TriggerDagRunOperator(
        task_id='trigger_train_dag',
        trigger_dag_id='train_dag',
        reset_dag_run=False,
        wait_for_completion=False,
    )

    get_application_data_task = get_application_data() 

    get_application_data_task >> upload_to_s3 >> trigger_train_dag