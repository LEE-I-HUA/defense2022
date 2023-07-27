from datetime import datetime, timedelta
from gc import collect
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
import time as tm
from pymongo import MongoClient
from pymongo import errors

import configparser
from bson import json_util
import os
import re
import spacy
import itertools
from bson.objectid import ObjectId
from io import StringIO
from sklearn.feature_extraction.text import CountVectorizer

from discordwebhook import Discord
# airflow 設定
default_args = {
	'owner': 'mirdc',
	'start_date': datetime(2023, 6, 5), # 代表從神麼時候開始第一次執行此 DAG
	'retries': 2, #  則允許 Airflow 在 DAG 失敗時重試 2 次
	'retry_delay': timedelta(minutes=0.08) # DAG 失敗後等多久後開始重試
}

homepath = '/home/mirdc/airflow/'

with DAG('test_R', default_args=default_args) as dag:
	
	create_rdata_task = BashOperator(
		task_id='create_rdata_task',
		bash_command=f'Rscript /home/mirdc/airflow/scripts/taskA.r',
		dag=dag,
	)
	
	# define workflow
	create_rdata_task