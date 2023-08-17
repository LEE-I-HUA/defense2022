import pandas as pd
from dash import Dash, html
from dash import dcc, no_update
from dash.dependencies import Input, Output, State
import dash_cytoscape as cyto # pip install dash-cytoscape
import numpy as np
import dash_bootstrap_components as dbc
import gc
#import dash_bootstrap_components as dbc

import visdcc # pip install visdcc
# In[]
# 字典
origin_key_dict_pd = pd.read_csv('./NER_old/entityDict.csv')
#origin_key_dict_pd = pd.read_csv('./NER_old/entityDict.csv')
# 關鍵字類別
keyword_class_list = ["com", "rocket", "org", "satellite", "term", "loc"]
filter_class_list = ["不篩選","com", "rocket", "org", "satellite", "term", "loc"]
#filter_class_list = ["com", "rocket", "org", "satellite", "term", "loc"]
# 類別顏色
color_list = ['rgb(141, 211, 199)','rgb(247, 129, 191)','rgb(190, 186, 218)','rgb(251, 128, 114)','rgb(146, 208, 80)','rgb(253, 180, 98)']
Sen_Doc_list = ["Sentence", "Document"]
# In[]
#lemma = pd.read_csv('./NER_old/doc_label_table.csv')
X = pd.read_csv('./NER_old/doc_raw_data.csv')
#raw_S = pd.read_csv('./new_data/sen_raw_data.csv')
XX_Sent = pd.read_csv('./NER_old/SenDTM.csv')
XX_Doc = pd.read_csv('./NER_old/DocDTM.csv')
senlabel = pd.read_csv('./NER_old/sen_label_table.csv')
raw_S = pd.read_csv('./NER_old/sen_raw_data.csv')
#coo_df = pd.read_csv('./new_data/DocCO_format.csv')
CR_doc = pd.read_csv('./NER_old/DocCR.csv')#CR_doc2 = pd.read_csv('./data/CRdoc1224.csv',index_col=0)
CR_sen = pd.read_csv('./NER_old/SenCR.csv')
CO_doc = pd.read_csv('./NER_old/DocCO.csv')
CO_sen = pd.read_csv('./NER_old/SenCO.csv')
# In[]
# 測試用
#Z = "Carnegie_Mellon_University"
#Z = "3D printing"  input_filter = ["com", "rocket", "loc"]
# Unit = "Sentence"  type = 'correlation' total_nodes_num = 10  threshold = 0.5 input_filter = "com"
# In[]
def calculate_edge_width(x, Min, Max):
    if Max - Min > 0:
        return (x - Min) / (Max - Min)
    else:
        return x

# 網路圖函數 Unit：計算單位(句、篇) Z：中心節點字 type：計算單位（CO或CR）total_nodes_num:網路圖節點數量 threshold:計算閥值 input_filter:篩選遮罩參數
def get_element_modify(Unit, Z, type, total_nodes_num, threshold, input_filter):
    
        node_size_list = []
        input_filter_list = []
        #value_list = []
        #print(input_filter)
        
        # 按照條件篩選資料     
        if type == 'correlation':
            if Unit == "Document":
                input_data = CR_doc
            elif Unit == "Sentence":
                input_data = CR_sen
            
     
            # 網路篩選遮罩
            if "不篩選" in input_filter:
                v = input_data[Z].tolist()#取出中心節點字每個值
                v = list(enumerate(v))#加上每個值加上索引
            
            elif(input_filter != None or input_filter != []):
                #for filter_item in input_filter:
                input_filter_list = [index for index, label in enumerate(origin_key_dict_pd['label']) if label not in input_filter]
                #input_filter_list = origin_key_dict_pd[origin_key_dict_pd['label'] != input_filter].index.tolist()#取出字典中符合該關鍵字'Label'所有索引
                v = [(index, input_data.loc[index, Z]) for index in input_filter_list]#取出資料裡符合Z和input_filter_list的值和索引
                            
            #else:
                #v = input_data[Z].tolist()#取出中心節點字每個值
                #v = list(enumerate(v))#加上每個值加上索引


            v = sorted(v, key=lambda x: x[1], reverse=True)#為每個值進行降序排列
            
            v_index = [i for i, _ in v][:total_nodes_num]#取出前K個索引
            col_index = [((input_data.columns).tolist())[i] for i in v_index]#獲取對應的欄位名
            x = input_data.loc[v_index, col_index]#根據v_index,col_index，分別做為欄和列索引取值
            x.columns = v_index
            del v
            gc.collect()  
            
            x_values = x.values# 獲取x的數據部分，轉換為numpy數組
            # 獲取下三角部分的boolean *x_values.shape:使用x_values數組的形狀來確定矩陣的行數和列數 dtype:設定矩陣資料型態 k:True或False比例
            lower_triangle = np.tri(*x_values.shape, dtype=bool, k=0)
            x_values[lower_triangle] = 0# 將下三角部分（True）的元素設置為0
            x_updated = pd.DataFrame(x_values, index=x.index, columns=x.columns)# 將更新後的numpy數組重新轉換為DataFrame
            del x
            gc.collect()  
            
            melted_df = x_updated.stack().reset_index()#轉成對應關係
            melted_df.columns = ['from', 'to', 'Value']#欄位命名
            melted_df = melted_df[melted_df['Value'] > 0].reset_index(drop=True)#找大於0的值
            melted_df[['from', 'to']] = np.sort(melted_df[['from', 'to']], axis=1)#按['from', 'to']排序
            melted_df = melted_df.drop_duplicates(subset=['from', 'to']).reset_index(drop=True)#刪除重複值
            
         
            value_list = melted_df["Value"].tolist()
            percentile = np.percentile(value_list, (threshold*100))#根據value_list算出閥值
            
            melted_df_thres = melted_df[melted_df['Value'] >= percentile].reset_index(drop=True)#取符合threshold的value
            melted_df_thres["Value"] = np.sqrt(melted_df_thres['Value'])#取平方根值
            del melted_df
            gc.collect()  
            
            #新增['from_name','to_name','id']的欄位，值為透過索引映射到對應值
            melted_df_thres['from_name'] = melted_df_thres['from'].map(dict(zip(v_index, col_index)))
            melted_df_thres['to_name'] = melted_df_thres['to'].map(dict(zip(v_index, col_index)))
            melted_df_thres['id'] = melted_df_thres['from_name'].astype(str) + "_" + melted_df_thres['to_name'].astype(str)

                       
            Min, Max = melted_df_thres['Value'].min(), melted_df_thres['Value'].max()#Value欄位Max,Min值
            #melted_df_thres['edge_width'] = melted_df_thres['Value'].apply(lambda x: ((x - Min) / (Max - Min)))#edge的寬度計算
            melted_df_thres['edge_width'] = melted_df_thres['Value'].apply(lambda x: calculate_edge_width(x, Min, Max))
            
            nodes_list = melted_df_thres['from_name'].tolist() + melted_df_thres['to_name'].tolist()
            nodes_list = list(set(nodes_list))#刪除重複值
            
            #取出字典中對應節點的freq值
            for node in nodes_list:
                node_size_list.append(int(origin_key_dict_pd[origin_key_dict_pd['keywords'] == node]['freq'].to_string().split()[1]))
            
            #用以計算節點大小
            size_total = sum(node_size_list)
        
        # 按照條件篩選資料
        elif type == 'co-occurrence':
            if Unit == "Document":
                input_data = CO_doc
                choose_data = CR_doc    
            elif Unit == "Sentence":
                input_data = CO_sen
                choose_data = CR_sen
            
            # 網路篩選遮罩
            if "不篩選" in input_filter:
                v = choose_data[Z].tolist()#取出中心節點字每個值
                v = list(enumerate(v))#加上每個值加上索引

            elif(input_filter != None or input_filter != []):
                #for filter_item in input_filter:
                input_filter_list = [index for index, label in enumerate(origin_key_dict_pd['label']) if label not in input_filter]
                #input_filter_list = origin_key_dict_pd[origin_key_dict_pd['label'] != input_filter].index.tolist()#取出字典中符合該關鍵字'Label'所有索引
                v = [(index, choose_data.loc[index, Z]) for index in input_filter_list]#取出資料裡符合Z和input_filter_list的值和索引

                
            v = sorted(v, key=lambda x: x[1], reverse=True)#為每個值進行降序排列
            
            v_index = [i for i, _ in v][:total_nodes_num]#取出前K個索引
            col_index = [((input_data.columns).tolist())[i] for i in v_index]#獲取對應的欄位名
            x = input_data.loc[v_index, col_index]#根據v_index,col_index，分別做為欄和列索引取值
            x.columns = v_index
            del v
            gc.collect() 
            
            x_values = x.values# 獲取x的數據部分，轉換為numpy數組
            # 獲取下三角部分的boolean *x_values.shape:使用x_values數組的形狀來確定矩陣的行數和列數 dtype:設定矩陣資料型態 k:True或False比例
            lower_triangle = np.tri(*x_values.shape, dtype=bool, k=0)
            x_values[lower_triangle] = 0# 將下三角部分（包括對角線）的元素設置為0
            x_updated = pd.DataFrame(x_values, index=x.index, columns=x.columns)# 將更新後的numpy數組重新轉換為DataFrame
            del x
            gc.collect() 
            
            melted_df = x_updated.stack().reset_index()#轉成對應關係
            melted_df.columns = ['from', 'to', 'Value']#欄位命名
            melted_df = melted_df[melted_df['Value'] > 0].reset_index(drop=True)#找大於0的值
            melted_df[['from', 'to']] = np.sort(melted_df[['from', 'to']], axis=1)#按['from', 'to']排序
            melted_df = melted_df.drop_duplicates(subset=['from', 'to']).reset_index(drop=True)#刪除重複值
            
            
            value_list = melted_df["Value"].tolist()
            percentile = np.percentile(value_list, (threshold*100))#根據value_list算出閥值
            
            melted_df_thres = melted_df[melted_df['Value'] >= percentile].reset_index(drop=True)#取符合threshold的value
            melted_df_thres["Value"] = np.sqrt(melted_df_thres['Value'])#取平方根值
            del melted_df
            gc.collect() 
            
            #新增['from_name','to_name','id']的欄位，值為透過索引映射到對應值
            melted_df_thres['from_name'] = melted_df_thres['from'].map(dict(zip(v_index, col_index)))
            melted_df_thres['to_name'] = melted_df_thres['to'].map(dict(zip(v_index, col_index)))
            melted_df_thres['id'] = melted_df_thres['from_name'].astype(str) + "_" + melted_df_thres['to_name'].astype(str)

            
            Min, Max = melted_df_thres['Value'].min(), melted_df_thres['Value'].max()#Value欄位Max,Min值
            #melted_df_thres['edge_width'] = melted_df_thres['Value'].apply(lambda x: ((x - Min) / (Max - Min)))#edge的寬度計算
            melted_df_thres['edge_width'] = melted_df_thres['Value'].apply(lambda x: calculate_edge_width(x, Min, Max))
            
            nodes_list = melted_df_thres['from_name'].tolist() + melted_df_thres['to_name'].tolist()
            nodes_list = list(set(nodes_list))#刪除重複值
            
            #取出字典中對應節點的freq值
            for node in nodes_list:
                node_size_list.append(int(origin_key_dict_pd[origin_key_dict_pd['keywords'] == node]['freq'].to_string().split()[1]))
                
            #用以計算節點大小    
            size_total = sum(node_size_list)
            
        # group:節點字類別 title:網路圖tooltip shape:節點形狀 size:節點大小          
        nodes = [
                {
                'id': node, 
                'label': node, 
                'group': origin_key_dict_pd[origin_key_dict_pd['keywords'] == node]['label'].to_string().split()[1],
                'title': node + ":({},{})".format(origin_key_dict_pd[origin_key_dict_pd['keywords'] == node]['label'].to_string().split()[1],origin_key_dict_pd[origin_key_dict_pd['keywords'] == node]['freq'].to_string().split()[1]),
                'shape': 'dot',
                'size': 15  + (int(origin_key_dict_pd[origin_key_dict_pd['keywords'] == node]['freq'].to_string().split()[1])/size_total)*100,
                }
                for node in nodes_list
                ]
        
        # width:邊寬度 title:網路圖tooltip
        edges = [
                {
                'id' : row['from_name']+'_'+row['to_name'], 
                'from': row['from_name'], 
                'to': row['to_name'], 
                'classes': type, #cor or coo
                'weight': row['Value'], 
                'width':row['edge_width']*6,
                'title': row['from_name']+'_'+row['to_name']
                 }        
                for idx, row in melted_df_thres[(melted_df_thres['from_name'].isin(nodes_list) & 
                                                                    melted_df_thres['to_name'].isin(nodes_list))].iterrows()
                 ]
                                                                      

        info = { "Unit": str(Unit),
                 "type": str(type),
                 "total_nodes_num": total_nodes_num,
                 "threshold": threshold
            }                                                                          
                                                                                  
        data ={'nodes':nodes,
               'edges':edges,
               'info': info
           }
        
        return data
                                
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
    'columns':[{'title': 'Date',
                'dataIndex': 'Date',
                'key': 'Date',
                'width': '20%'},
               {'title': 'doc_id',
                'dataIndex': 'id',
                'key': 'id',
                'width': '20%'},
                {'title': 'Recent',
                'dataIndex': 'Recent',
                'key': 'Recent',
                'width': '60%'},
                #{'title': 'url',
                #'dataIndex': 'url',
                #'key': 'url',
                #'width': '15%'}],
]    
}

app.layout = html.Div([
    html.Div([
        html.P("國防 太空文集 NER單中心網路分析", style={'font-size': '30px'}),
        html.H6('以特定主題為中心，從文集中選出相關性最高的關鍵詞，並對它們進行社會網絡分析',
                style={'color': 'Green',
                       'font-size': '12px',
                       }),
        ## 切換 class 下拉式選單
        dbc.Label("選擇關鍵字類別", style={'font-size': '12px'}),
        dcc.Dropdown(
            id='dropdown_choose_class',
            value= 4,
            clearable=False,
            options=[
                {'label': clas, 'value': i}
                for i, clas in enumerate(keyword_class_list)
            ]
        ),
        ## 選擇中心詞
        dbc.Label("選擇關鍵字", style={'font-size': '12px'}),
        dcc.Dropdown(
            id='dropdown_choose_name',
            value= '3D printing',
            clearable=False,
            options=[
                    {'label': name, 'value': name}
                    for name in origin_key_dict_pd[origin_key_dict_pd['label'] == keyword_class_list[0]]['keywords'].to_list()
            ]
        ),
        ## 網路篩選遮罩
        dbc.Label("網路篩選遮罩", style={'font-size': '12px'}),
        dcc.Dropdown(
            id='dropdown_choose_filter',
            value= "不篩選",
            clearable=False,
            multi=True,
            options=[
                {'label': method, 'value': method}
                for i, method in enumerate(filter_class_list)
            ]
        ),        
        html.H6('針對網路圖的節點可以進行篩選',
                style={'color': 'Green'}),
        dbc.Label("設定網路節點數量", style={'font-size': '12px'}),
        dcc.Slider(
                id="total_nodes_num_slider", min=4, max=20,step=1,
                marks={i: str(i) for i in range(21)},
                value=8
        ),
        dbc.Label("依關聯節度篩選鏈結", style={'font-size': '12px'}),
        dcc.Slider(
                id="threshold_slide", min=0, max=1,step=0.01,
                marks={i/10: str(i/10) for i in range(51)},
                value=0.5
        ),
        html.H6('如果字詞出現頻率較高，可以選擇「相關係數」來定義連結強度；如果字詞出現頻率較低，可以選擇「共同出現次數」作為連結強度',
                style={'color': 'Green'}),
        dbc.Label("字詞連結段落", style={'font-size': '12px'}),
        dcc.RadioItems(id='RadioItems_SenorDoc',
                       options=[{'label': '句 ', 'value': 'Sentence'},
                                {'label': '篇', 'value': 'Document'},],
                       value='Sentence',
                       inline=True,
                       ),
        dbc.Label("連結強度計算方式", style={'font-size': '12px'}),
        dcc.RadioItems(id='RadioItems_CRorCO',
                       options=[{'label': '共同出現次數'  , 'value': 'co-occurrence'},
                                {'label': '相關係數', 'value': 'correlation'},],
                       value='correlation',
                       inline=False,
                       ),

    ],style = {'height' : 850 ,'width': '20%', 'display': 'inline-block','backgroundColor':'rgb(232, 237, 248)'}),
    html.Div([
        # 網路圖Legend
        dcc.Markdown('''
                ![Legend](https://i.ibb.co/s6RL68v/Legend0716.png)       
        ''',
        style={'width': '60%', 'height': 20}
        ),
        # 網路圖
        visdcc.Network(
                id='net',
                selection={'nodes': [], 'edges': []},
                options={
                    'interaction':{
                        'hover': True,
                        'tooltipDelay': 300,
                        },
                    'groups':{
                        'com': {'color':'rgb(251, 128, 114)'},
                        'rocket': {'color':'rgb(253, 180, 98)'},
                        'org': {'color':'rgb(190, 186, 218)'},
                        'satellite': {'color':'rgb(247, 129, 191)'},
                        'term': {'color':'rgb(141, 211, 199)'},
                        'loc': {'color':'rgb(146, 208, 80)'},
                        },
                    'autoResize': True,
                    'height': '800px',
                    'width': '100%',
                    'layout': {
                        'improvedLayout':True,
                        'hierarchical': {
                          'enabled':False,
                          'levelSeparation': 150,
                          'nodeSpacing': 100,
                          'treeSpacing': 200,
                          'blockShifting': True,
                          'edgeMinimization': True,
                          'parentCentralization': True,
                          'direction': 'UD',        # UD, DU, LR, RL
                          'sortMethod': 'hubsize'   # hubsize, directed
                        }
                    },
                    'physics':{
                         'enabled': True,
                         'barnesHut': {
                                  'theta': 0.5,
                                  'gravitationalConstant': -20000,#repulsion強度
                                  'centralGravity': 0.3,
                                  'springLength': 95,
                                  'springConstant': 0.04,
                                  'damping': 0.09,
                                  #'avoidOverlap': 0.01
                                },
                },
                    'adaptiveTimestep': True,
                }
            
                        ),
            
                    
    ],style = {'height' : '100%' ,'width': '60%', 'display': 'inline-block'}),
    
    html.Div([
        # 文本元件
        dcc.Textarea(
            id='textarea-example',
            #   value='paragraph',
            style={'width': '100%', 'height': 350},
            disabled = True,
        ),
        html.Div([
            # 資料表
            visdcc.DataTable(
                id         = 'table' ,
                box_type   = 'radio',
                style={'width': '100%', 'height': 500},
                data       = table_data
            ),
        ])
    ],style = {'height' : '150%' ,'width': '35%', 'display': 'inline-block'}),
    
],style = {'height' : '100%' , 'display': 'flex'})


## 切換 class 下拉式選單
@app.callback(
    Output("dropdown_choose_name", 'options'),
    Input("dropdown_choose_class", "value"),
)
def update_elements(class_idx):#當dropdown_choose_class下拉選單的值發生變化時，會觸發，class_idx類別索引
    ## 選擇中心詞
    options=[
                {'label': name, 'value': name}
                for name in origin_key_dict_pd[origin_key_dict_pd['label'] == keyword_class_list[class_idx]]['keywords'].to_list()
            ]
    
    return options


#更新下拉選單
@app.callback(
    Output("threshold_slide", 'min'),
    Output("threshold_slide", 'max'),
    Output("threshold_slide", 'marks'),
    Output("threshold_slide", 'value'),
    Input("RadioItems_CRorCO", 'value')
)
def update_elements(type):
    # if type == 'correlation':
    min=0
    max=1
    marks={i/10: str(i/10) for i in range(11)}
    value=0.3
        
    if type == 'co-occurrence':
        min=0
        max=1
        marks={i/10: str(i/10) for i in range(11)}
        value=0.3

    return min, max, marks, value

#當dropdown-update-layout下拉選單的值發生變化時，更新網路圖
@app.callback(
    Output("net", 'data'),
    Input('RadioItems_SenorDoc', 'value'),
    Input("dropdown_choose_name", 'value'),
    Input("total_nodes_num_slider", "value"),
    Input('RadioItems_CRorCO', 'value'),
    Input('threshold_slide', 'value'),
    Input('dropdown_choose_filter', 'value'),
)
def update_elements(Unit,center_node, total_nodes_num, type, threshold, input_filter):
    
    return get_element_modify(Unit,center_node, type, total_nodes_num, threshold, input_filter)

# In[]
# 測試用
#data = ['NASA']
# In[]
def node_recation(Unit, data, type, total_nodes_num, threshold):
    
    k = data[0]#所點擊的node值
    v = XX_Sent[k]#取關鍵詞矩陣
    v = np.where(v == 1)[0]#矩陣中值為1的索引
    v = v.tolist()

    index = raw_S.loc[v]#透過索引取值
    merged_df = pd.merge(index, senlabel, on=['doc_id', 'sen_id'])#透過'doc_id', 'sen_id'進行合併
    merged_df = pd.merge(merged_df, X, on='doc_id', how='left')#透過'doc_id'進行合併
    merged_df = merged_df.drop_duplicates(subset=['doc_id', 'sen_id'], keep='first').reset_index(drop=True)
    #merged_df[['doc_id', 'sen_id']] = np.sort(merged_df[['doc_id', 'sen_id']], axis=1)
    #merged_df = merged_df.drop_duplicates(subset=['doc_id', 'sen_id']).reset_index(drop=True)
    merged_df['date'] = pd.to_datetime(merged_df['date']).dt.date
    merged_df = merged_df.sort_values(by='date', ascending=False).reset_index(drop=True)
    
    if len(merged_df) > 1000:
        merged_df = merged_df[:999]
        
    #merged_df['artDate_Url'] = merged_df.apply(lambda row: f'<a href="{row["artUrl"]}">{row["artDate"]}</a>', axis=1)
    merged_df['artDate_Url'] = merged_df.apply(lambda row: html.A(html.P(row['date']), href=row['link']), axis=1)
    
    return merged_df, k
    
# In[]
# 測試用
#data = ['ABL_UK']
#from_token = "3D printing"
#to_token = "ArianeGroup"
# In[]
def edge_recation(Unit, data, type, total_nodes_num, threshold):
    
    # 將資料分為from,to token
    from_to_token =  data[0].split("_")    
    from_token = from_to_token[0] 
    to_token = from_to_token[1]
     

    if Unit == "Sentence":        
        token_df = XX_Sent[[from_token, to_token]]#透過from_token和to_token取關鍵詞矩陣
        token_df['total'] = token_df[from_token] + token_df[to_token]
        token_df = token_df[(token_df[from_token] == 1) & (token_df[to_token] == 1)]#取出為1的值
        
        index = raw_S.loc[token_df.index.tolist()]#透過前面的index獲取sen_raw_data.csv的值
        merged_df2 = pd.merge(index, senlabel, on=['doc_id', 'sen_id'])# 以['doc_id', 'sen_id']兩欄與sen_label_table.csv執行Merge
        merged_df2 = pd.merge(merged_df2, X, on='doc_id', how='left')# 和原始資料進行Merge
        merged_df2 = merged_df2.drop_duplicates(subset=['doc_id', 'sen_id'], keep='first').reset_index(drop=True)#刪除重複值
        
        #將['artDate']資料型態轉為datatime後，降序排列
        merged_df2['date'] = pd.to_datetime(merged_df2['date']).dt.date
        merged_df2 = merged_df2.sort_values(by='date', ascending=False).reset_index(drop=True)
        
    else:        
        token_df = XX_Sent[[from_token, to_token]]#透過from_token和to_token取關鍵詞矩陣
        token_df['total'] = token_df[from_token] + token_df[to_token]
        token_df = token_df[token_df['total'] >= 1]#取出['total']大於等於1的值
        
        index = raw_S.loc[token_df.index.tolist()]#透過前面的index獲取sen_raw_data.csv的值
        merged_df2 = pd.merge(index, senlabel, on=['doc_id', 'sen_id'])# 以['doc_id', 'sen_id']兩欄與sen_label_table.csv執行Merge
        merged_df2 = pd.merge(merged_df2, X, on='doc_id', how='left')# 和原始資料進行Merge
        merged_df2 = merged_df2.drop_duplicates(subset=['doc_id', 'sen_id'], keep='first').reset_index(drop=True)#刪除重複值
        
        #將['artDate']資料型態轉為datatime後，降序排列
        merged_df2['date'] = pd.to_datetime(merged_df2['date']).dt.date
        merged_df2 = merged_df2.sort_values(by='date', ascending=False).reset_index(drop=True)
    
    #避免datatable元件過載
    if len(merged_df2) > 1000:
        merged_df2 = merged_df2[:999]
        
    #將artDate加入超連結功能，測試中        
    merged_df2['artDate_Url'] = merged_df2.apply(lambda row: html.A(html.P(row['date']), href=row['link']), axis=1)
 
    
    return merged_df2, from_token, to_token


# Datatable更新函數
@app.callback(
    Output('table', 'data'),
    Input('RadioItems_SenorDoc', 'value'),
    Input('net', 'selection'),
    Input("total_nodes_num_slider", "value"),
    Input('RadioItems_CRorCO', 'value'),
    Input('threshold_slide', 'value'),
)
def update_elements(Unit, selection, total_nodes_num, type, threshold):
    global merged_df
    res = []
    
    if len(selection['nodes']) != 0:
        #print(selection)
        #將node對應資料映射到datatable
        merged_df, token = node_recation(Unit, selection['nodes'], total_nodes_num, type, threshold)
        for i, j, k, l in zip(merged_df['date'], merged_df['doc_id'], merged_df['ner_sen'], merged_df['link']):
            res.append({'Date':i, 'id':j, 'Recent':k, 'url':l})
        table_data['columns'] = [
            {'title': 'Date',
            'dataIndex': 'Date',
            'key': 'Date',
            'width': '20%'},
            {'title': 'doc_id',
            'dataIndex': 'id',
            'key': 'id',
            'width': '20%'},
            {'title': 'Recent:{}({})'.format(token,len(merged_df)),
            'dataIndex': 'Recent',
            'key': 'Recent',
            'width': '60%'},
            #{'title': 'url',
            #'dataIndex': 'url',
            #'key': 'url',
            #'width': '15%'}
        ]
    elif len(selection['edges']) != 0:
        #print(selection)
        #將edge對應資料映射到datatable
        merged_df2, from_token, to_token = edge_recation(Unit, selection['edges'], total_nodes_num, type, threshold)
        for i, j, k, l in zip(merged_df2['date'], merged_df2['doc_id'], merged_df2['ner_sen'], merged_df2['link']):
            res.append({'Date':i, 'id':j, 'Recent':k, 'url':l})
        table_data['columns'] = [
            {'title': 'Date',
            'dataIndex': 'Date',
            'key': 'Date',
            'width': '20%'},
            {'title': 'doc_id',
            'dataIndex': 'id',
            'key': 'id',
            'width': '20%'},
            {'title': 'Recent:{}({})'.format(from_token + "_" + to_token,len(merged_df2)),
            'dataIndex': 'Recent',
            'key': 'Recent',
            'width': '60%'},
            #{'title': 'url',
            #'dataIndex': 'url',
            #'key': 'url',
            #'width': '15%'}
        ]
    else:
        table_data['columns'] = [
            {'title': 'Date',
            'dataIndex': 'Date',
            'key': 'Date',
            'width': '20%'},
            {'title': 'doc_id',
            'dataIndex': 'id',
            'key': 'id',
            'width': '20%'},
            {'title': 'Recent',
            'dataIndex': 'Recent',
            'key': 'Recent',
            'width': '60%'},
            #{'title': 'url',
            #'dataIndex': 'url',
            #'key': 'url',
            #'width': '15%'}
        ]
        
    table_data['dataSource'] = res

    return table_data

# textarea更新函數
@app.callback(
    Output('textarea-example', 'value'),
    Input('table', 'box_selected_keys')
)
def myfun(box_selected_keys): 
    #print([box_selected_keys[0]])
    if box_selected_keys == None:
        return ''
    else: 
        return merged_df['ner_doc'][box_selected_keys[0]]
    


app.run_server(debug=True, use_reloader=False)  # Turn off reloader if inside Jupyter