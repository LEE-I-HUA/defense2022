import pandas as pd
from dash import Dash, html
from dash import dcc, no_update
from dash.dependencies import Input, Output, State
import dash_cytoscape as cyto # pip install dash-cytoscape
import numpy as np
import dash_bootstrap_components as dbc
import gc
from datetime import date
from datetime import datetime
#import dash_bootstrap_components as dbc
import math
import spacy
import itertools
import os
import re
import ast
from sklearn.feature_extraction.text import CountVectorizer
 
import visdcc # pip install visdcc
import emoji

import sys
sys.setrecursionlimit(1500)
# In[] 
entityDict = pd.read_csv('./NER_old/entityDict.csv')
#entityDict = pd.read_csv('/new_data/NER_new/entityDict.csv')
doc_kw_data = pd.read_csv('./new_data/NER_new/doc_kw_data.csv', index_col = 0)
doc_label_table = pd.read_csv('./new_data/NER_new/doc_label_table.csv')

doc_kw_data['doc_kw_list'] = doc_kw_data['doc_kw_list'].apply(lambda x: eval(x))
doc_kw_data = doc_kw_data.explode('doc_kw_list')
n_clicks_counter = 1
# In[] 
# 外部css元件
external_stylesheets = ['https://unpkg.com/antd@3.1.1/dist/antd.css',
                        'https://rawgit.com/jimmybow/CSS/master/visdcc/DataTable/Filter.css',
                        'https://cdnjs.cloudflare.com/ajax/libs/vis/4.20.1/vis.min.css',
                        ]

app = Dash(__name__, external_stylesheets = external_stylesheets)


styles = {
    'pre': {
        'border': 'thin lightgrey solid',#邊框屬性
        'overflowX': 'scroll'#內容水平延伸
    }
}

merged_df = pd.DataFrame()

table_data = {
    'dataSource':[],
    'columns':[
                {'title': 'Token',
                'dataIndex': 'Token',
                'key': 'Token',
                'width': '100%'},
               #{'title': 'doc_id',
                #'dataIndex': 'id',
                #'key': 'id',
                #'width': '20%'},
                #{'title': 'Recent',
                #'dataIndex': 'Recent',
                #'key': 'Recent',
                #'width': '60%'},
                #{'title': 'url',
                #'dataIndex': 'url',
                #'key': 'url',
                #'width': '15%'}],
]    
}

app.layout = html.Div(children=[
    html.H1("國防太空文集", 
        style={
            'font-size': '36px',
            'textAlign': 'center',
            'backgroundColor': '#daf5ed',
            'margin': '0px',
            'font-weight': 'bold',
            'padding': '5px'
            }
        ),
    html.Div([
        html.Div([
            ## 選擇中心詞
            dbc.Label("輸入自訂關鍵字", 
                      style={
                          'font-size': '16px',
                          'color': '#CA774B',
                          'font-weight': 'bold'
                          }
                      ),
            dcc.Input(id='input_text', 
                      type='text', 
                      #value='HELSINKI', 
                      style={'whiteSpace': 'pre-wrap'},
                      ),
            html.Br(),
            html.Button(id='submit_button', n_clicks=1, children='Submit',style={'whiteSpace': 'pre-wrap'}),
            ], 
            style = {
                #'height' : 2000,
                #'height' : 2000,
                'height': 500,
                #'width': '20%', 
                'width': '15%',
                'display': 'inline-block',
                #'backgroundColor':'rgb(210, 238, 229)',
                'background-color': '#daf5ed',
                'padding': '0.5%'
                }
            ),
        html.Div([
            visdcc.DataTable(
                id         = 'table' ,
                #box_type   = 'radio',
                style={
                    'width': '100%', 
                    'height': 500,
                        'background-color': "#66828E",
                        'color': 'white',
                        'padding': '0.5%',
                        #'verticalAlign': 'top'
                    },
                data       = table_data
            ),  
            ],
            style = {
                #'height' : '150%',
                'width': '85%', 
                'background-color': "#66828E",
                'color': 'white',
                'padding': '0.5%',
                'verticalAlign': 'top',
                'display': 'inline-block',
                #'display': 'compact',
                }
            )
        ])
    ], 
    style = {})


# Datatable更新函數
@app.callback(
    Output('table', 'data'),
    Input('submit_button', 'n_clicks'),
    State('input_text', 'value'),
)
def datatable_show(n_clicks, input_text):
    global n_clicks_counter
    global doc_kw_data
    global entityDict
    res = []
    
    if n_clicks == n_clicks_counter:
        for i in zip(doc_kw_data['doc_kw_list']):
            res.append({'Token': str(i)})
        table_data['columns'] = [
                {'title': 'Token',
                'dataIndex': 'Token',
                'key': 'Token',
                'width': '100%'},
                #{'title': 'doc_id',
                #'dataIndex': 'id',
                #'key': 'id',
                #'width': '20%'},
                #{'title': 'Recent:{}({})'.format(token,len(merged_df)),
                #'dataIndex': 'Recent',
                #'key': 'Recent',
                #'width': '60%'},
                #{'title': 'url',
                #'dataIndex': 'url',
                #'key': 'url',
                #'width': '15%'}
            ]
        
    elif n_clicks > n_clicks_counter:
        
        for i in zip(doc_kw_data['doc_kw_list']):
            res.append({'Token': str(i)})
        table_data['columns'] = [
                {'title': 'Token',
                'dataIndex': 'Token',
                'key': 'Token',
                'width': '100%'},
                #{'title': 'doc_id',
                #'dataIndex': 'id',
                #'key': 'id',
                #'width': '20%'},
                #{'title': 'Recent:{}({})'.format(token,len(merged_df)),
                #'dataIndex': 'Recent',
                #'key': 'Recent',
                #'width': '60%'},
                #{'title': 'url',
                #'dataIndex': 'url',
                #'key': 'url',
                #'width': '15%'}
        ]
            
        input_text_list = input_text.split(',')
        
        #print(input_text_list[0])
        #print(input_text_list[1])
        
        new_keyword = input_text_list[0]
        new_keyword_class = input_text_list[1]
        #entityDict_keywords_list = entityDict['keywords'].tolist()
        new_entityDict_row = pd.DataFrame({'keywords':new_keyword,
                               'label': new_keyword_class,
                               'word_label_count': 100,
                               'freq': 100,},
                                          index=[ 0 ]
                                          )
        
        if new_keyword not in entityDict['keywords'].tolist():
            
            
            entityDict = pd.concat([entityDict, new_entityDict_row])
            #entityDict = entityDict.append({'keywords': new_keyword, 'label': new_keyword_class, 'word_label_count': 100, 'freq': 100}, ignore_index=False)
            entityDict.to_csv('./new_data/NER_new/entityDict.csv', index=False,encoding='utf-8')
    

        
    table_data['dataSource'] = res
    
    #print(res)
    
    return table_data #entityDict
   
    
app.run_server(debug=True, use_reloader=False)  # Turn off reloader if inside Jupyter
