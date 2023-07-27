from datetime import datetime, timedelta
from gc import collect
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
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

def discord_chatbot(discord_link, msg):
    discord = Discord(url=discord_link)
    discord.post(content=msg)

dis_link = 'https://discord.com/api/webhooks/1093224027960651829/wU8Gz76_kUNsq92Et1Ng6S2Ab6MMDwYcvE9XzsAkQOYG4f0n9yVxHx_1wwSNb5Eedyc0'
# Get current directory
homepath = '/home/mirdc/airflow/'

# airflow 設定
default_args = {
	'owner': 'mirdc',
	'start_date': datetime(2023, 4, 20), # 代表從神麼時候開始第一次執行此 DAG
	'retries': 2, #  則允許 Airflow 在 DAG 失敗時重試 2 次
	'retry_delay': timedelta(minutes=0.08) # DAG 失敗後等多久後開始重試
}

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36'}
pre_href = 'https://spacenews.com/wp-json/newspack-blocks/v1/articles?className=is-style-borders&moreButton=1&showAvatar=0&postsToShow=500&mediaPosition=right&categories%5B0%5D='
mid_herf = '&typeScale=5&imageScale=2&mobileStack=1&showExcerpt=1&excerptLength=55&showReadMore=0&readMoreLabel=Keep%20reading&showDate=1&showImage=1&showCaption=0&disableImageLazyLoad=0&imageShape=landscape&minHeight=0&moreButtonText&showAuthor=1&showCategory=0&postLayout=list&columns=3&colGap=3&&&&&&sectionHeader&specificMode=0&textColor&customTextColor&singleMode=0&showSubtitle=0&postType%5B0%5D=post&textAlign=left&includedPostStatuses%5B0%5D=publish&page='

news_code = '4161'
opinion_code = '4163'

def connect_db():
	# connect to mongodb
	# config = configparser.ConfigParser()
	# config.read(homepath+'dags/data/config.ini')

	# MongoDB = config["MongoDB"]["Database"]
	# MongoUser = config["MongoDB"]["User"]
	# MongoPW = config["MongoDB"]["PW"]
	# MongoURL = config["MongoDB"]["URL"]

	# uri = "mongodb://" + MongoUser + ":" + MongoPW + "@"+ MongoURL + "/" + MongoDB + "?authMechanism=SCRAM-SHA-1"
    
	url = 'mongodb://localhost:27017/'
	client = MongoClient(url)
	db = client.airflowDB
	collection = db.spacenews

	return collection
	

#全部的url
def get_urls(opinion_code, last_link):    
    url_list = []
    page = 0 
    turn = True    #控制是否要繼續抓下一頁

    while turn:
        url = pre_href + opinion_code + mid_herf +str(page)
        print(url)
        res = requests.get(url, headers=headers)
        for key in res.json()["items"]:
            information = key["html"]
            newsHerf = re.search("https.*/", information).group(0)
            if newsHerf == last_link:
                turn = False
                break
            else:
                url_list.append(newsHerf)  
        page += 1
        tm.sleep(0.5)
    print('有',len(url_list),'筆新資料')
    return url_list

#文章內容
def get_content(url,category):   
	segment = []
	tag = []
	content = ''
	title=''
	author=''
	date=''

	res = requests.get(url,headers)
	soup = BeautifulSoup(res.text,'html.parser')
	try:
		main_content = soup.find('div', class_= 'main-content')
		title = main_content.find('h1', class_="entry-title").getText().strip()
		author = main_content.find('span', class_='byline').getText().replace('\nby\n','')
		date = main_content.find('span', class_='posted-on').time.get('datetime')
		date = re.sub('-\d+:\d+','',date)

		for line in main_content.find('article').find('div',class_='entry-content').select('p'):
			content+=line.getText()
			content+='\n'

		for line in main_content.find('footer', class_="entry-footer").select('a'):
			tag.append(line.getText().strip())
		
	except Exception as error:
		print(url,':',error)

	result={
		'link':url,
		'title': title,
		'author': author,
		'date': date,
		'segment': segment,
		'tag': tag,
		'content': content,
		'subject': category,
		'html': res.text
	}
	return result

def get_last_link():
	collection = connect_db()
	with collection.find({'subject':'news'}).sort("date", -1).limit(1) as docs:
		news_last_link = docs[0]['link']
			
	with collection.find({'subject':'opinion'}).sort("date", -1).limit(1) as docs:
		opinion_last_link = docs[0]['link']
	return news_last_link, opinion_last_link


def news_get_url(**context):
	last_date_news = context['task_instance'].xcom_pull(task_ids='get_last_link_task')[0]
	# news part
	urls_news = get_urls(news_code,last_date_news)

	print('news有',len(urls_news),'筆資料')
	return urls_news


def news_get_content(**context):
	urls_news = context['task_instance'].xcom_pull(task_ids='news_get_url_task')
	content_list = []
	for i in urls_news:
		content_news = get_content(i,'news')
		content_list.append(content_news)
	# content_json = json.dumps(content_list, default=json_util.default)

	return content_list

def opinion_get_url(**context):
	last_opinion_link = context['task_instance'].xcom_pull(task_ids='get_last_link_task')[1]
	#opinion part
	urls_opinion = get_urls(opinion_code,last_opinion_link)
	print('opinion有',len(urls_opinion),'筆資料')
	return urls_opinion


def opinion_get_content(**context):
	urls_opinion = context['task_instance'].xcom_pull(task_ids='opinion_get_url_task')
	content_list = []

	for i in urls_opinion:
		content_opinion = get_content(i,'opinion')
		content_list.append(content_opinion)
	# content_json = json.dumps(content_list, default=json_util.default)

	return content_list

#將文章內容新增至資料庫
def news_insert_database(**context):
	content_list = context['task_instance'].xcom_pull(task_ids='news_get_content_task')
	for doc in content_list:
		if doc['date']!='':
			doc['date'] = datetime.strptime(doc['date'], "%Y-%m-%dT%H:%M:%S")

	# content_list = json.loads(content_json)
	if len(content_list)==0:
		print('爬蟲news沒有更新資料')
	else:
		collection = connect_db()
		collection.insert_many(content_list)
		print('爬蟲更新',len(content_list),'筆資料')
		msg = f"爬蟲news更新{len(content_list)}筆資料"
		discord_chatbot(dis_link, msg)


#將文章內容新增至資料庫
def opinion_insert_database(**context):
	content_list = context['task_instance'].xcom_pull(task_ids='opinion_get_content_task')
	for doc in content_list:
		if doc['date']!='':
			doc['date'] = datetime.strptime(doc['date'], "%Y-%m-%dT%H:%M:%S")
	# content_list = json.loads(content_json)
	if len(content_list)==0:
		print('爬蟲沒有更新資料')
	else:
		collection = connect_db()
		collection.insert_many(content_list)
		print('爬蟲更新',len(content_list),'筆資料')
		msg = f"爬蟲opinion更新{len(content_list)}筆資料"
		discord_chatbot(dis_link, msg)
		
# 做文字處理
def token_replace():
	collection = connect_db()
	origin_data = pd.DataFrame(collection.find({ 'tokens' : { '$exists': False } },{'content':1}))
	new_data = []

	if len(origin_data)>0:
		origin_data['content'] = origin_data['content'].str.replace('\n|\t',' ')
		origin_data['content'] = origin_data['content'].str.replace('  ',' ')
		print('需要更新斷詞的文章數：',len(origin_data))
		contentDF = origin_data

		# 匯入字典
		Dict = pd.read_csv(homepath+'dags/data/Defense_entity_dict - space.csv')

		# 將字典裡的entity作為要被替換後的字
		Dict['replacement'] = Dict['entity'].str.replace('-','_')
		Dict['replacement'] = Dict['replacement'].str.replace(',','')
		Dict['replacement'] = Dict['replacement'].str.replace('\.','')
		Dict['replacement'] = Dict['replacement'].str.replace('\'','')
		Dict['replacement'] = Dict['replacement'].str.replace(' ','_')

		#有些字前面會斷不乾淨，會變成").Seed_Innovations_LLC"這樣，因此在前面加空白
		Dict.loc[Dict['replacement']=='Seed_Innovations_LLC','replacement'] = ' Seed_Innovations_LLC '
		Dict.loc[Dict['replacement']=='Ram_Photonics_LLC','replacement'] = ' Ram_Photonics_LLC '
		Dict.loc[Dict['replacement']=='C_UAS','replacement'] = ' C_UAS '
		Dict.loc[Dict['replacement']=='5G','replacement'] = '5_G'

		# 字典依照字的長度排序
		for i in range(len(Dict)):
			Dict.loc[i,'len'] = len(Dict.loc[i,'tooltip'])
			
		Dict = Dict.sort_values(by='len', ascending=False)

		contentDF['doc_id'] = contentDF.index
		
		for index, row in contentDF.iterrows():
			text = row['content']
			for index_D , row_D in Dict.iterrows():
				text = re.sub(row_D['alias'], row_D['replacement'], text, flags = re.I)
			new_data.append({'_id':row['_id'],'doc_id':row['doc_id'],'content':text})

	else:
		print('需要更新斷詞的文章數：0')
	return json.loads(json_util.dumps(new_data))


# 斷詞
def tokenize(**context):
	total_new_data = context['task_instance'].xcom_pull(task_ids='token_replace_task')

	result=[]
	if len(total_new_data) == 0:
		print('不須更新')
	else:
		contentDF = pd.json_normalize(total_new_data).rename(columns={'_id.$oid':'_id'})
		nlp = spacy.load("en_core_web_md") # python -m spacy download en_core_web_md

		contentDF['index_'] = contentDF._id # 將index獨立成一個名為index_的欄位
		data = contentDF

		gen1, gen2 = itertools.tee(data.itertuples())
		ids = (doc.index_ for doc in gen1)
		# texts = (doc.Title for doc in gen2)
		texts = (doc.content for doc in gen2)

		# 要超過最長的文章長度
		nlp.max_length = 100000
		index = 0
		docus = nlp.pipe(texts, batch_size=100, n_process=3)
		for id_, doc in zip(ids, docus):
			sentence_result = []
			token_result = []
			for sent_id, sent in enumerate(doc.sents):
				for token in sent:
					# 清除空白、標點符號
					if not token.is_space and not token.is_punct:
						token_result.append({"sen_id":sent_id+1,"token": token.text,"lemma": token.lemma_,'POS':token.pos_})
				sentence_result.append({"sen_id":sent_id+1,"sen":sent.text})
			result.append({"_id":id_, 'sentence':sentence_result, 'tokens':token_result})
	return result


def update_DB_nlpData(**context):
	result = context['task_instance'].xcom_pull(task_ids='tokenize_task')
	collection = connect_db()
	for doc in result:
		db_filter = { '_id':ObjectId(doc['_id'])}

		sentence_values = { "$set": { 'sentence':doc['sentence']} }
		tokens_values = { "$set": { 'tokens':doc['tokens']} }

		collection.update_one(db_filter, sentence_values)
		collection.update_one(db_filter, tokens_values)
		
	print(len(result),'筆斷詞資料更新')

# 製作dtm
def dtm_pre_process():
	collection = connect_db()
	print('從資料庫撈資料')
	get_date = datetime.now() - timedelta(days=365*3)
	# total_data = pd.DataFrame(collection.find({'date':{'$gte': get_date}},
    #                     {'_id':0,'tokens':1}))
	total_data = pd.DataFrame(collection.find({},
                        {'tokens':1,'sentence':1}))
	total_data = total_data.sort_values('_id')
	total_data = total_data.reset_index(drop=True)
	total_data = total_data.drop(columns=['_id'])
	total_data['doc_id'] = total_data.index+1

	sen_data = total_data.explode('sentence').reset_index(drop=True)
	sen_data['sen_id'] = sen_data['sentence'].apply(lambda x: x['sen_id'])
	sen_data = sen_data.drop(columns = ['sentence','tokens'])

	print('製作DTM前處理')
	start = datetime.now()
	token_data = total_data.explode('tokens')
	token_data['sen_id'] = token_data['tokens'].apply(lambda x: x['sen_id'])
	token_data['token'] = token_data['tokens'].apply(lambda x: x['token'])
	token_data = token_data.drop(columns = ['sentence','tokens'])

	Dict = pd.read_csv('/home/mirdc/airflow/dags/data/Defense_entity_dict - space.csv')
	Dict['entity'] = Dict['entity'].str.replace('-','_')
	Dict['entity'] = Dict['entity'].str.replace(',','')
	Dict['entity'] = Dict['entity'].str.replace('\.','')
	Dict['entity'] = Dict['entity'].str.replace('\'','')
	Dict['entity'] = Dict['entity'].str.replace(' ','_')
	Dict.loc[Dict['entity']=='5G','entity'] = '5_G'
	Dict = Dict.drop_duplicates(subset=['entity'])

	# 找出有出現在文章中的關鍵字
	kw_tokens =  token_data.loc[token_data['token'].isin(Dict['entity'])]
	del token_data
	# 以句子為單位找出有出現關鍵字的文章句子組合
	sen_kw_DF = kw_tokens.groupby(['doc_id','sen_id'])['token'].apply(list).reset_index(name='kw_tokens')
	# 和完整的文章句子合併
	sen_tokens = sen_data.merge(sen_kw_DF, how='left')
	sen_tokens.loc[sen_tokens['kw_tokens'].isna(), 'kw_tokens']='[]'
	del sen_kw_DF

	doc_kw_DF = kw_tokens.groupby('doc_id')['token'].apply(list).reset_index(name='kw_tokens')
	del kw_tokens
	doc_tokens = total_data.merge(doc_kw_DF, how='left')
	del doc_kw_DF,total_data
	doc_tokens.loc[doc_tokens['kw_tokens'].isna(), 'kw_tokens']='[]'

	sen_tokens = sen_tokens.sort_values(by=['doc_id','sen_id'])
	doc_tokens = doc_tokens.sort_values(by='doc_id')

	end = datetime.now()
	print('花費時間：', (end-start)/60)
	print('製作句子 DTM')
	start = datetime.now()

	docs = [" ".join(token) for token in sen_tokens['kw_tokens']]
	# 計算詞頻
	vec = CountVectorizer(vocabulary=Dict['entity'].unique(),lowercase=False)

	# data in form of csr_matrix
	X = vec.fit_transform(docs)

	# data in form of pandas.DataFrame
	SenDTM = pd.DataFrame(X.todense())
	SenDTM.columns = vec.get_feature_names_out()
	SenDTM = SenDTM.fillna(0)
	SenCR = SenDTM.corr()
	SenDTM.to_csv('/home/mirdc/airflow/dags/data/SenDTM.csv',index = False)
	del SenDTM

	SenCR = SenCR.fillna(0)
	SenCR.to_csv('/home/mirdc/airflow/dags/data/SenCR.csv',index = False)
	del SenCR

	SenCO = (X.T * X) # matrix manipulation
	SenCO.setdiag(0)
	names = vec.get_feature_names_out() # This are the entity names (i.e. keywords)
	SenCO = pd.DataFrame(data = SenCO.toarray(), columns = names, index = names)
	SenCO.to_csv('/home/mirdc/airflow/dags/data/SenCO.csv',index = False)
	del SenCO

	end = datetime.now()
	print('花費時間：', (end-start)/60)

	print('製作文章 DTM')
	start = datetime.now()

	docs = [" ".join(token) for token in doc_tokens['kw_tokens']]
	# 計算詞頻
	vec = CountVectorizer(vocabulary=Dict['entity'].unique(),lowercase=False)

	# data in form of csr_matrix
	X = vec.fit_transform(docs)

	# data in form of pandas.DataFrame
	DocDTM = pd.DataFrame(X.todense())
	DocDTM.columns = vec.get_feature_names_out()
	DocDTM = DocDTM.fillna(0)
	DocDTM.to_csv('/home/mirdc/airflow/dags/data/DocDTM.csv',index = False)
	DocCR = DocDTM.corr()
	del DocDTM

	DocCR = DocCR.fillna(0)
	DocCR.to_csv('/home/mirdc/airflow/dags/data/DocCR.csv',index = False)
	del DocCR

	DocCO = (X.T * X) # matrix manipulation
	DocCO.setdiag(0)
	names = vec.get_feature_names_out() # This are the entity names (i.e. keywords)
	DocCO = pd.DataFrame(data = DocCO.toarray(), columns = names, index = names)
	DocCO.to_csv('/home/mirdc/airflow/dags/data/DocCO.csv',index = False)
	del DocCO

	Dict.to_csv('/home/mirdc/airflow/dags/data/new_dict.csv',index = False)
	
	end = datetime.now()
	print('花費時間：', (end-start)/60)
	msg = f'更新完畢\n更新時間:{ (end-start)/60}分鐘'
	discord_chatbot(dis_link,msg)

with DAG('Spacenews', default_args=default_args,schedule_interval='0 17 * * *') as dag:
	# get database newest time
	get_last_link_task = PythonOperator(
		task_id = 'get_last_link_task',
		python_callable = get_last_link
	)
	# get news url
	news_get_url_task = PythonOperator(
		task_id = 'news_get_url_task',
		python_callable = news_get_url
	)
	# get news content
	news_get_content_task = PythonOperator(
		task_id = 'news_get_content_task',
		python_callable = news_get_content
	)
	# insert news content to db
	news_insert_database_task = PythonOperator(
		task_id = 'news_insert_database_task',
		python_callable = news_insert_database
	)


	# get opinion content
	opinion_get_url_task = PythonOperator(
		task_id = 'opinion_get_url_task',
		python_callable = opinion_get_url
	)
	opinion_get_content_task = PythonOperator(
		task_id = 'opinion_get_content_task',
		python_callable = opinion_get_content
	)
	# insert opinion content to db
	opinion_insert_database_task = PythonOperator(
		task_id = 'opinion_insert_database_task',
		python_callable = opinion_insert_database
	)

	token_replace_task = PythonOperator(
		task_id = 'token_replace_task',
		python_callable = token_replace
	)

	tokenize_task = PythonOperator(
		task_id = 'tokenize_task',
		python_callable = tokenize
	)

	dtm_pre_process_task = PythonOperator(
		task_id = 'dtm_pre_process_task',
		python_callable = dtm_pre_process
	)

	update_DB_nlpData_task = PythonOperator(
		task_id = 'update_DB_nlpData_task',
		python_callable = update_DB_nlpData
	)

	create_rdata_task = BashOperator(
		task_id='create_rdata_task',
		bash_command=f'{homepath}scripts/run_r.sh {homepath}scripts/build_R_data.r',
		dag=dag,
	)
	
	NER_task= TriggerDagRunOperator(
        trigger_dag_id="Spacenews_NER",
        task_id='tokenize_ner_task'
    )

	# define workflow
	get_last_link_task >> [news_get_url_task, opinion_get_url_task]
	news_get_url_task >> news_get_content_task >> news_insert_database_task
	opinion_get_url_task >> opinion_get_content_task >> opinion_insert_database_task
	[news_insert_database_task ,  opinion_insert_database_task] >> token_replace_task >> tokenize_task >> update_DB_nlpData_task >> dtm_pre_process_task >> create_rdata_task
	create_rdata_task >> NER_task