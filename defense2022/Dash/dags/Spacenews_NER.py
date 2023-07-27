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
import ast

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

from spacy.training import biluo_tags_to_offsets
from spacy.training import iob_to_biluo
from spacy.training import biluo_tags_to_spans

nlp = spacy.load("en_core_web_md")

stopwords = list(nlp.Defaults.stop_words)
stopwords = stopwords + ['','to','In','long','not','value','It','science','wind','design','0.15','1,100','150','you','will']
stopwords = stopwords+['12','15','50','9/11','d0','cut','old','Q-','Q.','sea','sky','1,000','air','all','All','and','Are','As','big']


def clean_entity(text):
    text = re.sub("'s$", '', text)
    text = re.sub("’s$", '', text)
    text = re.sub('”', '', text)
    text = re.sub('\)', '', text)
    text = re.sub('\(', '', text)
    if len(text)<2:
        text = ''
    return(text)

def discord_chatbot(discord_link, msg):
    discord = Discord(url=discord_link)
    discord.post(content=msg)

dis_link = 'https://discord.com/api/webhooks/1093224027960651829/wU8Gz76_kUNsq92Et1Ng6S2Ab6MMDwYcvE9XzsAkQOYG4f0n9yVxHx_1wwSNb5Eedyc0'
# Get current directory
homepath = '/home/mirdc/airflow/'

# airflow 設定
default_args = {
	'owner': 'mirdc',
	'start_date': datetime(2023, 7, 17), # 代表從神麼時候開始第一次執行此 DAG
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
	
def dummy(doc):
    return doc

# 斷詞
def tokenize_ner():
	collection = connect_db()
	need_ner_token = pd.DataFrame(collection.find({ 'ner_token' : { '$exists': False } },{'content':1}))
	# need_ner_token = pd.DataFrame(collection.find({},{'content':1}))[10:15].reset_index(drop=True)
	old_doc_id = pd.DataFrame(collection.find({ 'doc_id' : { '$exists': True } },{'doc_id':1,'date':1}).sort([('doc_id', -1)]).limit(1))['doc_id'].iloc[0]
	# old_doc_id = 9
	print('old_doc_id:',old_doc_id)
	need_ner_token['doc_id'] = old_doc_id + 1 + need_ner_token.index

	doc_sen_list = []
	doc_sen_token_list = []
	doc_list = []

	if len(need_ner_token) == 0:
		print('不須更新')
	else:
		need_ner_token['content'] = need_ner_token['content'].str.replace('‘',' ')
		need_ner_token['content'] = need_ner_token['content'].str.replace('’',' ')
		need_ner_token['content'] = need_ner_token['content'].str.replace('\s{2,}','')
		need_ner_token['content'] = need_ner_token['content'].str.replace('\n','')
		need_ner_token['content'] = need_ner_token['content'].str.replace('\xa0','')
		need_ner_token['content'] = need_ner_token['content'].str.replace('“','')
		need_ner_token['content'] = need_ner_token['content'].str.replace('”','')
		
		nlp = spacy.load("en_core_web_md") # python -m spacy download en_core_web_md

		need_ner_token['index_'] = need_ner_token.doc_id # 將index獨立成一個名為index_的欄位

		gen1, gen2 = itertools.tee(need_ner_token.itertuples())
		ids = (doc.index_ for doc in gen1)
		# texts = (doc.Title for doc in gen2)
		texts = (doc.content for doc in gen2)

		# 要超過最長的文章長度
		nlp.max_length = 100000
		docus = nlp.pipe(texts, batch_size=100, n_process=3)

		for id_, doc in zip(ids, docus):
			sentence_token_list = []
			sent_list = []
			doc_txt = ''
			for sent_id, sent in enumerate(doc.sents):
				token_list = []
				for token in sent:
					token_list.append(token.text)
				sent_list.append(sent.text)
				sentence_token_list.append(token_list)
			doc_list.append(doc.text)
			doc_sen_list.append(sent_list)
			doc_sen_token_list.append(sentence_token_list)
			

		for doc_id, doc in enumerate(doc_sen_token_list):
			new_doc_id = old_doc_id+1+doc_id
			file_name = '/home/mirdc/DeepKE/example/ner/few-shot/data/spacenews/doc' + str(new_doc_id) + '.txt'
			with open(file_name,'w') as f:
				for sen in doc:
					for token in sen:
						text = token + '	' + 'B-COM'
						f.write(text)
						f.write('\n')
					f.write('\n')
	return doc_list, doc_sen_list, doc_sen_token_list

def update_DB_nerData(**context):
# def update_DB_nerData(doc_list, doc_sen_list, doc_sen_token_list):
	doc_list = context['task_instance'].xcom_pull(task_ids='tokenize_ner_task')[0]
	doc_sen_list = context['task_instance'].xcom_pull(task_ids='tokenize_ner_task')[1]
	doc_sen_token_list = context['task_instance'].xcom_pull(task_ids='tokenize_ner_task')[2]
	
	collection = connect_db()
	need_ner_token = pd.DataFrame(collection.find({ 'ner_token' : { '$exists': False } },{'content':1}))
	old_doc_id = pd.DataFrame(collection.find({ 'doc_id' : { '$exists': True } },{'doc_id':1,'date':1}).sort([('doc_id', -1)]).limit(1))['doc_id'].iloc[0]
	need_ner_token['doc_id'] = old_doc_id + 1 + need_ner_token.index

	for index, row in need_ner_token.iterrows():
		db_filter = { '_id': ObjectId(row['_id']) }
		doc_id = row['doc_id']
		ner_doc = doc_list[index]

		ner_doc_sen = doc_sen_list[index]
		sen_dict = []
		for idx, sent in enumerate(ner_doc_sen):
			sen_dict.append({'sen_id':idx, 'ner_sen':sent})
		

		ner_doc_sen_token = doc_sen_token_list[index]
		token_dict = []
		for idx, sent in enumerate(ner_doc_sen_token):
			for tokens in sent:
				token_dict.append({'sen_id':idx, 'ner_token':tokens})

		try:
			new_doc_id = { "$set": { 'doc_id':  doc_id} }
			collection.update_one(db_filter, new_doc_id)

			new_ner_doc = { "$set": { 'ner_doc':  ner_doc} }
			collection.update_one(db_filter, new_ner_doc)

			new_ner_doc_sen = { "$set": { 'ner_sen':  sen_dict} }
			collection.update_one(db_filter, new_ner_doc_sen)

			new_ner_doc_sen_token = { "$set": { 'ner_token':  token_dict} }
			collection.update_one(db_filter, new_ner_doc_sen_token)

		except Exception as error:
			print(index)
			print(error)
	
	print(len(need_ner_token),'筆斷詞資料更新')

# LightNER 結果
def processing_LightNER():
	result_sen = []
	result_doc = []

	file_list = sorted(os.listdir('/home/mirdc/DeepKE/example/ner/few-shot/outputs/spacenews'))
	if len(file_list)==0:
		print('不用更新')
	else:
		first_doc_id =  int(re.findall('\d+',file_list[0])[0])

		## 原文
		### 文章
		collection = connect_db()
		new_ner_data = pd.DataFrame(collection.find({'doc_id':{'$gte':first_doc_id}},{'doc_id','link','title','date','subject','ner_doc','ner_sen'}))
		# new_ner_data = pd.DataFrame(collection.find({},{}))
		doc_raw_data = new_ner_data[['doc_id','link','title','date','subject','ner_doc']]
		doc_raw_data.to_csv('/home/mirdc/ShinyApps/defense/data/NER_new/doc_raw_data.csv', index=False, encoding='utf-8')

		### 句子
		sen_raw_data = new_ner_data[['doc_id','ner_sen']].explode(['ner_sen'])
		sen_raw_data['sen_id'] = sen_raw_data['ner_sen'].str['sen_id']
		sen_raw_data['ner_raw_sen'] = sen_raw_data['ner_sen'].str['ner_sen']
		sen_raw_data = sen_raw_data.drop(columns=['ner_sen']).reset_index(drop=True)
		sen_raw_data = sen_raw_data.rename(columns={'ner_raw_sen':'ner_sen'})
		sen_raw_data.to_csv('/home/mirdc/ShinyApps/defense/data/NER_new/sen_raw_data.csv', index=False, encoding='utf-8')

		for file in file_list:
			filename = '/home/mirdc/DeepKE/example/ner/few-shot/outputs/spacenews/' + file
			doc_id = int(re.findall('\d+',file)[0])
			print('doc_id',doc_id)
			with open(filename, newline='', encoding="utf-8") as f:     
				lines = f.readlines()
			a_sen_label_list = []
			a_doc_label_list = []
			soc_sen_label_list = []

			for line in lines[2:]:
				if line != '\n':
					token_label  = line.split()
					if len(token_label) < 2:
						token = ' '
						label = 'O'
					else:
						token = token_label[0]
						label = token_label[1]
					a_sen_label_list.append(label)
					a_doc_label_list.append(label)
				# 換句了
				else:
					soc_sen_label_list.append(a_sen_label_list)
					a_sen_label_list = []
			doc_tags_u = iob_to_biluo(a_doc_label_list)

			doc_nlp = nlp(doc_raw_data.loc[doc_raw_data['doc_id'] == doc_id]['ner_doc'].iloc[0])

			entities = biluo_tags_to_offsets(doc_nlp, doc_tags_u)
			doc_kw_list = biluo_tags_to_spans(doc_nlp, doc_tags_u)
			result_doc.append({'doc_enties':entities,  'doc_kw':[clean_entity(i.text) for i in doc_kw_list]})

			doc_sen_text = sen_raw_data.loc[sen_raw_data['doc_id'] == doc_id]
			sen_entities = []
			for t in range(len(doc_sen_text)):
				sen_nlp = nlp(doc_sen_text.loc[sen_raw_data['sen_id'] == t]['ner_sen'].iloc[0])
				tags = iob_to_biluo(soc_sen_label_list[t])
				entities = biluo_tags_to_offsets(sen_nlp, tags)
				sen_kw_list = biluo_tags_to_spans(sen_nlp, tags)
				sen_entities.append({'sen_id':t,'entities':entities, 'sen_kw':[clean_entity(i.text) for i in sen_kw_list]})
			result_sen.append(sen_entities)

			## move predict data to other Folder
			os.rename(filename, '/home/mirdc/DeepKE/example/ner/few-shot/outputs/old/' + file)
			input_file = '/home/mirdc/DeepKE/example/ner/few-shot/data/spacenews/' + re.sub('predict_','',file)
			os.rename(input_file, '/home/mirdc/DeepKE/example/ner/few-shot/data/old/' + re.sub('predict_','',file))
			
	return result_doc, result_sen
	
def lightNER_keywords_doc(**context):
	result_doc = context['task_instance'].xcom_pull(task_ids='processing_LightNER_task')[0]
	if len(result_doc):
		print('不用更新')
	else:
		doc_raw_data = pd.read_csv('/home/mirdc/ShinyApps/defense/data/NER_new/doc_raw_data.csv')
		doc_kw_data = doc_raw_data.assign(ner_result_doc = result_doc)
		doc_kw_data['doc_kw_list'] = doc_kw_data['ner_result_doc'].str['doc_kw']
		doc_kw_data[['doc_id','doc_kw_list']].to_csv('/home/mirdc/ShinyApps/defense/data/NER_new/doc_kw_data.csv', index=False, encoding='utf-8')

		doc_raw_data = doc_raw_data[['doc_id','ner_doc']]
		doc_label_table = doc_raw_data.assign(ner_result_doc = result_doc)
		doc_label_table['doc_entities_list'] = doc_label_table['ner_result_doc'].str['doc_enties']
		doc_label_table['doc_kw_list'] = doc_kw_data['ner_result_doc'].str['doc_kw']
		doc_label_table = doc_label_table.explode(['doc_entities_list','doc_kw_list']).drop(columns=['ner_result_doc'])
		doc_label_table['start'] = doc_label_table['doc_entities_list'].str[0]
		doc_label_table['end'] = doc_label_table['doc_entities_list'].str[1]
		doc_label_table['label'] = doc_label_table['doc_entities_list'].str[2]
		doc_label_table = doc_label_table.drop(columns=['doc_entities_list','ner_doc']).reset_index(drop=True)

		ent_labelDF = doc_label_table.loc[doc_label_table['doc_kw_list'] != '']
		ent_labelDF = ent_labelDF.sort_values('doc_kw_list').drop(columns=['start','end','doc_id']).rename(columns={'doc_kw_list':'keywords'})

		ent_freq = pd.DataFrame({'freq':ent_labelDF.groupby('keywords').size()}).reset_index()
		ent_labelDF['word_label_count'] = ent_labelDF.groupby(['keywords','label']).cumcount()
		ent_labelDF = ent_labelDF.sort_values('word_label_count')
		entityDict = ent_labelDF.drop_duplicates(['keywords'],keep='last')
		entityDict['word_label_count']  = entityDict['word_label_count']+1
		entityDict = entityDict.merge(ent_freq)

		entityDict = entityDict.loc[entityDict['keywords'].isin(stopwords)==False]
		old_entityDict = pd.read_csv('/home/mirdc/ShinyApps/defense/data/NER_old/entityDict.csv')
		newDict = pd.concat([old_entityDict, entityDict]).groupby(['keywords', 'label']).sum().reset_index()
		newDict = newDict.loc[newDict['freq']>=100]
		newDict = newDict.sort_values('freq').drop_duplicates(['keywords'],keep='last')
		newDict = newDict.sort_values('keywords')
		newDict.to_csv('/home/mirdc/ShinyApps/defense/data/NER_old/entityDict.csv', index=False,encoding='utf-8')
		print(newDict.groupby('label').size())
		print(len(newDict))

		# 將doc_label_table去除停用字、錯誤的類別更正
		doc_label_table = doc_label_table.loc[doc_label_table['doc_kw_list'].isin(entityDict['keywords'])]
		doc_label_table = doc_label_table.drop(columns=['label']).merge(newDict[['keywords','label']],left_on='doc_kw_list',right_on='keywords').drop(columns=['keywords'])

		doc_label_table.to_csv('/home/mirdc/ShinyApps/defense/data/NER_new/doc_label_table.csv', index=False,encoding='utf-8')

	
def lightNER_keywords_sen(**context):
	result_sen = context['task_instance'].xcom_pull(task_ids='processing_LightNER_task')[1]
	if len(result_sen):
		print('不用更新')
	else:
		entityDict = pd.read_csv('/home/mirdc/ShinyApps/defense/data/NER_old/entityDict.csv')
	
		doc_raw_data = pd.read_csv('/home/mirdc/ShinyApps/defense/data/NER_new/doc_raw_data.csv')
		sen_kw_data = doc_raw_data[['doc_id']].assign(ner_result_sen = result_sen)
		sen_kw_data = sen_kw_data.explode('ner_result_sen')
		sen_kw_data['sen_kw_list'] = sen_kw_data['ner_result_sen'].str['sen_kw']
		sen_kw_data['sen_id'] = sen_kw_data['ner_result_sen'].str['sen_id']
		sen_kw_data.drop(columns=['ner_result_sen']).to_csv('/home/mirdc/ShinyApps/defense/data/NER_new/sen_kw_data.csv', index=False, encoding='utf-8')


		sen_label_table = sen_kw_data
		sen_label_table['sen_entities_list'] = sen_label_table['ner_result_sen'].str['entities']
		sen_label_table['sen_id'] = sen_label_table['ner_result_sen'].str['sen_id']

		sen_label_table = sen_label_table.explode(['sen_kw_list','sen_entities_list']).drop(columns=['ner_result_sen'])
		sen_label_table['start'] = sen_label_table['sen_entities_list'].str[0]
		sen_label_table['end'] = sen_label_table['sen_entities_list'].str[1]
		sen_label_table['label'] = sen_label_table['sen_entities_list'].str[2]
		sen_label_table = sen_label_table.drop(columns=['sen_entities_list']).reset_index(drop=True)

		## 將doc_label_table去除停用字、錯誤的類別更正
		sen_label_table = sen_label_table.loc[sen_label_table['sen_kw_list'].isin(entityDict['keywords'])]
		sen_label_table = sen_label_table.drop(columns=['label']).merge(entityDict[['keywords','label']],left_on='sen_kw_list',right_on='keywords').drop(columns=['keywords'])

		sen_label_table.to_csv('/home/mirdc/ShinyApps/defense/data/NER_new/sen_label_table.csv', index=False,encoding='utf-8')

def renew_csv_data():
    doc_raw_data_old = pd.read_csv('/home/mirdc/ShinyApps/defense/data/NER_old/doc_raw_data.csv')
    doc_raw_data_new = pd.read_csv('/home/mirdc/ShinyApps/defense/data/NER_new/doc_raw_data.csv')
    doc_raw_data = pd.concat([doc_raw_data_old, doc_raw_data_new]).reset_index(drop=True)
    doc_raw_data = doc_raw_data.drop_duplicates(subset=['doc_id'])
    doc_raw_data.to_csv('/home/mirdc/ShinyApps/defense/data/NER_old/doc_raw_data.csv',index=False, encoding='utf-8')

    sen_raw_data_old = pd.read_csv('/home/mirdc/ShinyApps/defense/data/NER_old/sen_raw_data.csv')
    sen_raw_data_new = pd.read_csv('/home/mirdc/ShinyApps/defense/data/NER_new/sen_raw_data.csv')
    sen_raw_data = pd.concat([sen_raw_data_old, sen_raw_data_new]).reset_index(drop=True)
    sen_raw_data = sen_raw_data.drop_duplicates(subset=['doc_id','sen_id'])
    sen_raw_data.to_csv('/home/mirdc/ShinyApps/defense/data/NER_old/sen_raw_data.csv',index=False, encoding='utf-8')

    doc_kw_data_old = pd.read_csv('/home/mirdc/ShinyApps/defense/data/NER_old/doc_kw_data.csv')
    doc_kw_data_new = pd.read_csv('/home/mirdc/ShinyApps/defense/data/NER_new/doc_kw_data.csv')
    doc_kw_data = pd.concat([doc_kw_data_old, doc_kw_data_new]).reset_index(drop=True)
    doc_kw_data['doc_kw_list'] = doc_kw_data['doc_kw_list'].apply(lambda x: ast.literal_eval(x))
    doc_kw_data = doc_kw_data.drop_duplicates(subset=['doc_id'])
    doc_kw_data.to_csv('/home/mirdc/ShinyApps/defense/data/NER_old/doc_kw_data.csv',index=False, encoding='utf-8')

    sen_kw_data_old = pd.read_csv('/home/mirdc/ShinyApps/defense/data/NER_old/sen_kw_data.csv')
    sen_kw_data_new = pd.read_csv('/home/mirdc/ShinyApps/defense/data/NER_new/sen_kw_data.csv')
    sen_kw_data = pd.concat([sen_kw_data_old, sen_kw_data_new]).reset_index(drop=True)
    sen_kw_data['sen_kw_list'] = sen_kw_data['sen_kw_list'].apply(lambda x: ast.literal_eval(x))
    sen_kw_data = sen_kw_data.drop_duplicates(subset=['doc_id','sen_id'])
    sen_kw_data.to_csv('/home/mirdc/ShinyApps/defense/data/NER_old/sen_kw_data.csv',index=False, encoding='utf-8')

    sen_label_table_old = pd.read_csv('/home/mirdc/ShinyApps/defense/data/NER_old/sen_label_table.csv')
    sen_label_table_new = pd.read_csv('/home/mirdc/ShinyApps/defense/data/NER_new/sen_label_table.csv')
    sen_label_table = pd.concat([sen_label_table_old, sen_label_table_new]).reset_index(drop=True)
    sen_label_table = sen_label_table.drop_duplicates()
    sen_label_table.to_csv('/home/mirdc/ShinyApps/defense/data/NER_old/sen_label_table.csv',index=False, encoding='utf-8')

    doc_label_table_old = pd.read_csv('/home/mirdc/ShinyApps/defense/data/NER_old/doc_label_table.csv')
    doc_label_table_new = pd.read_csv('/home/mirdc/ShinyApps/defense/data/NER_new/doc_label_table.csv')
    doc_label_table = pd.concat([doc_label_table_old, doc_label_table_new]).reset_index(drop=True)
    doc_label_table = doc_label_table.drop_duplicates()
    doc_label_table.to_csv('/home/mirdc/ShinyApps/defense/data/NER_old/doc_label_table.csv',index=False, encoding='utf-8')
# 製作dtm
def NER_DTM_doc():

	doc_kw_data = pd.read_csv('/home/mirdc/ShinyApps/defense/data/NER_old/doc_kw_data.csv')
	doc_kw_data['doc_kw_list'] = doc_kw_data['doc_kw_list'].apply(lambda x: ast.literal_eval(x))
	entityDict = pd.read_csv('/home/mirdc/ShinyApps/defense/data/NER_old/entityDict.csv')
	# 計算詞頻
	vec = CountVectorizer(
			tokenizer=dummy,
			preprocessor=dummy,
			vocabulary=list(entityDict['keywords']))
	
	# data in form of csr_matrix
	X = vec.fit_transform(list(doc_kw_data['doc_kw_list']))

	# data in form of pandas.DataFrame
	DocDTM = pd.DataFrame(X.todense())
	DocDTM.columns = vec.get_feature_names_out()
	# DocDTM = DocDTM.drop(columns=[''])
	DocDTM = DocDTM.fillna(0)

	DocCO = (X.T * X) # matrix manipulation
	DocCO.setdiag(0)
	names = vec.get_feature_names_out() # This are the entity names (i.e. keywords)
	DocCO = pd.DataFrame(data = DocCO.toarray(), columns = names, index = names)

	DocCR = DocDTM.corr()
	DocCR = DocCR.fillna(0)
    
	DocDTM.to_csv('/home/mirdc/ShinyApps/defense/data/DocDTM.csv', index=False,encoding='utf-8')
	DocCO.to_csv('/home/mirdc/ShinyApps/defense/data/DocCO.csv', index=False,encoding='utf-8')
	DocCR.to_csv('/home/mirdc/ShinyApps/defense/data/DocCR.csv', index=False,encoding='utf-8')

# 製作dtm
def NER_DTM_sen():
	
	sen_kw_data_new = pd.read_csv('/home/mirdc/ShinyApps/defense/data/NER_old/sen_kw_data.csv')
	sen_kw_data_new['sen_kw_list'] = sen_kw_data_new['sen_kw_list'].apply(lambda x: ast.literal_eval(x))
	entityDict = pd.read_csv('/home/mirdc/ShinyApps/defense/data/NER_old/entityDict.csv')

	# 計算詞頻
	vec = CountVectorizer(
			tokenizer=dummy,
			preprocessor=dummy,
			vocabulary=list(entityDict['keywords']))
	# data in form of csr_matrix
	X = vec.fit_transform(list(sen_kw_data_new['sen_kw_list']))

	SenDTM = pd.DataFrame(X.todense())
	SenDTM.columns = vec.get_feature_names_out()
	# SenDTM = SenDTM.drop(columns=[''])
	SenDTM = SenDTM

	SenDTM = SenDTM.fillna(0)

	SenCO = (X.T * X) # matrix manipulation
	SenCO.setdiag(0)
	names = vec.get_feature_names_out() # This are the entity names (i.e. keywords)
	SenCO = pd.DataFrame(data = SenCO.toarray(), columns = names, index = names)

	SenCR = SenDTM.corr()
	SenCR = SenCR.fillna(0)

	SenDTM.to_csv('/home/mirdc/ShinyApps/defense/data/SenDTM.csv', index=False,encoding='utf-8')
	SenCO.to_csv('/home/mirdc/ShinyApps/defense/data/SenCO.csv', index=False,encoding='utf-8')
	SenCR.to_csv('/home/mirdc/ShinyApps/defense/data/SenCR.csv', index=False,encoding='utf-8')

with DAG('Spacenews_NER', default_args=default_args) as dag:
	## NER斷詞
	tokenize_ner_task = PythonOperator(
		task_id = 'tokenize_ner_task',
		python_callable = tokenize_ner
	)

	## update ner token
	update_DB_nerData_task = PythonOperator(
		task_id = 'update_DB_nerData_task',
		python_callable = update_DB_nerData
	)
	
	## NER Predict
	run_lightNER_task = BashOperator(
		task_id='run_lightNER_task',
		bash_command=f'{homepath}scripts/run_lightNER.sh ',
		dag=dag,
	)

	## NER output
	processing_LightNER_task = PythonOperator(
		task_id = 'processing_LightNER_task',
		python_callable = processing_LightNER
	)

	## doc kw
	lightNER_keywords_doc_task = PythonOperator(
		task_id = 'lightNER_keywords_doc_task',
		python_callable = lightNER_keywords_doc
	)

	## sen kw
	lightNER_keywords_sen_task = PythonOperator(
		task_id = 'lightNER_keywords_sen_task',
		python_callable = lightNER_keywords_sen
	)

	## renew csv
	renew_csv_data_task = PythonOperator(
		task_id = 'renew_csv_data_task',
		python_callable = renew_csv_data
	)

	## build doc DTM
	NER_DTM_doc_task = PythonOperator(
		task_id = 'NER_DTM_doc_task',
		python_callable = NER_DTM_doc
	)

	## build sen DTM
	NER_DTM_sen_task = PythonOperator(
		task_id = 'NER_DTM_sen_task',
		python_callable = NER_DTM_sen
	)

	## builde R data
	create_ner_rdata_task = BashOperator(
		task_id='create_ner_rdata_task',
		bash_command=f'{homepath}scripts/run_r.sh {homepath}scripts/build_NER_data.r',
		dag=dag,
	)


	# define workflow
	tokenize_ner_task >> update_DB_nerData_task >> run_lightNER_task >> processing_LightNER_task >> lightNER_keywords_doc_task >> lightNER_keywords_sen_task >> renew_csv_data_task
	renew_csv_data_task >> [NER_DTM_doc_task, NER_DTM_sen_task] >> create_ner_rdata_task