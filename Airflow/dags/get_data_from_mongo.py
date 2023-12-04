from airflow import DAG
from airflow.operators.python import PythonOperator
from pymongo import MongoClient
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 12, 5),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'mongo_dag',
    default_args=default_args,
    description='A simple Airflow DAG to read data from MongoDB using pymongo',
    schedule_interval=timedelta(days=1),
)

def get_data_from_mongo_atlas(ti):
    client = MongoClient('mongodb+srv://bruce1115:b3848948389!!@cluster0.c3fiz0r.mongodb.net/?retryWrites=true&w=majority')

    # Specify the database and collection
    db = client['test']  # Replace with your actual database name
    collection = db['users']  # Replace with your actual collection name
    
    query = {'role': 'passer'}
    result = collection.find(query)
    
    ti.xcom_push(key="passers", value=result)

    for document in result:
        print(document)

    # Close the MongoDB connection
    client.close()

with dag:
    task_read_mongo_data = PythonOperator(
        task_id='read_mongo_data',
        python_callable=get_data_from_mongo_atlas,
        provide_context=True,
    )

task_read_mongo_data

# def _get_url(ti):
#     pathlib.Path("/home/airflow/data").mkdir(parents=True, exist_ok=True)
#     TTBKey = 'Myapi_key'
#     items = []
#     for start_value in range(1, 11):
#         url = f"http://www.aladin.co.kr/ttb/api/ItemList.aspx?ttbkey={TTBKey}&QueryType=ItemNewAll&SearchTarget=Used&SubSearchTarget=Book&MaxResults=50&start={start_value}&output=js&Version=20131101&OptResult=usedList"
#         res = requests.get(url)
#         items.extend(res.json()['item'])
#     # items 리스트를 Airflow의 XCom 메커니즘을 통해 다른 작업과 공유 가능 이를 통해 items 값을 다른 작업에서 사용가능
#     ti.xcom_push(key="items", value=items)
    
# def insert_data_to_mongo_atlas(ti):
# 	# 이전 작업에서 XCom을 통해 전달된 데이터를 가져온다.
#     data = ti.xcom_pull(key="items")
#     # 자신의 MongoDB Atlas와 연결해준다. user와 password값 등록
#     client = MongoClient('mongodb+srv://user:<password>@cluster0.wydppxv.mongodb.net/?retryWrites=true&w=majority')
    
#     db = client['etl']
#     collection = db['aladin']
#     for item in data:
#         collection.insert_one(item)
    

# get_url = PythonOperator(
#     task_id="get_url", python_callable=_get_url, dag=dag
# )

# insert_task = PythonOperator(
#     task_id='insert_to_mongo_atlas',
#     python_callable=insert_data_to_mongo_atlas,
#     # 작업 함수에 실행 컨텍스트를 제공
#     #  ti 매개변수를 통해 TaskInstance를 사용
#     provide_context=True,
#     dag=dag,
# )
# get_url >>  insert_task