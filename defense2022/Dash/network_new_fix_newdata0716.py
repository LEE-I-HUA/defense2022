import pandas as pd
from dash import Dash, html
from dash import dcc, no_update
from dash.dependencies import Input, Output, State
import dash_cytoscape as cyto # pip install dash-cytoscape
import numpy as np
import dash_bootstrap_components as dbc
#import dash_bootstrap_components as dbc

import visdcc # pip install visdcc
# In[]
origin_key_dict_pd = pd.read_csv('./new_data/entityDict.csv')[["label", "keywords"]]#種類、名稱

keyword_class_list = ["com", "rocket", "org", "satellite", "term", "loc"]
filter_class_list = ["不篩選","com", "rocket", "org", "satellite", "term", "loc"]
Sen_Doc_list = ["Sentence", "Document"]
# In[]
lemma = pd.read_csv('./new_data/doc_label_table.csv')
X = pd.read_csv('./new_data/ner_data_bio.csv')
#raw_S = pd.read_csv('./new_data/sen_raw_data.csv')
XX_Sent = pd.read_csv('./new_data/SenDTM.csv')
XX_Doc = pd.read_csv('./new_data/DocDTM.csv')
senlabel = pd.read_csv('./new_data/sen_label_table.csv')
raw_S = pd.read_csv('./new_data/sen_raw_data.csv')
#coo_df = pd.read_csv('./new_data/DocCO_format.csv')
CR_doc = pd.read_csv('./new_data/DocCR.csv')#CR_doc2 = pd.read_csv('./data/CRdoc1224.csv',index_col=0)
CR_sen = pd.read_csv('./new_data/SenCR.csv')
CO_doc = pd.read_csv('./new_data/DocCO.csv')
CO_sen = pd.read_csv('./new_data/SenCO.csv')
# In[]
#color_list = ['#FB8072','#80B1D3','#BFB39B','#FDB462','#B3DE69','#FFFFB3']
color_list = ['rgb(141, 211, 199)','rgb(247, 129, 191)','rgb(190, 186, 218)','rgb(251, 128, 114)','rgb(146, 208, 80)','rgb(253, 180, 98)']
color_dict = dict(zip(keyword_class_list, color_list))
#Z = "Carnegie_Mellon_University"
#Z = "GPS"
#Z = Z.lower()
# Unit = "Sentence"  type = 'correlation' total_nodes_num = 10  threshold = 0.5 input_filter = "term"
def get_element_modify(Unit, Z, type, total_nodes_num, threshold, input_filter):
            
        if type == 'correlation':
            if Unit == "Document":
                input_data = CR_doc

            elif Unit == "Sentence":
                input_data = CR_sen
            
            if(input_filter != "不篩選"):
                input_filter_list = origin_key_dict_pd[origin_key_dict_pd['label'] == input_filter].index.tolist()#取出字典中符合該關鍵字'Label'所有索引
                #v = input_data[Z].tolist() 
                #v = input_data.loc[input_filter_list,Z].tolist()
                #v = list(set(v) & set(input_filter_list))
                v = [(index, input_data.loc[index, Z]) for index in input_filter_list]#取出資料裡符合Z和篩選遮罩(index)的值
            else:
                v = input_data[Z].tolist()#取出欄位Z的每個值
                v = list(enumerate(v))#加上每個值加上索引
                #v = list(set(v) & set(input_filter_list))
            
            #v = input_data[Z].tolist()
            #v = list(set(v) & set(input_filter_list))
            #v = list(enumerate(v))
            v = sorted(v, key=lambda x: x[1], reverse=True)#為每個值進行降序排列
            
            #v = list(set(list1) & set(list2))
            
            v_index = [i for i, _ in v][:total_nodes_num]#取出前K個索引
            col_index = [((input_data.columns).tolist())[i] for i in v_index]#獲取對應的欄位名
            x = input_data.loc[v_index, col_index]#根據v_index,col_index，分別做為欄和列索引取值
            x.columns = v_index
            
            x_values = x.values# 獲取x的數據部分，轉換為numpy數組
            lower_triangle = np.tri(*x_values.shape, dtype=bool, k=0)# 獲取下三角部分的boolean
            x_values[lower_triangle] = 0# 將下三角部分（包括對角線）的元素設置為0
            x_updated = pd.DataFrame(x_values, index=x.index, columns=x.columns)# 將更新後的numpy數組重新轉換為DataFrame

            
            melted_df = x_updated.stack().reset_index()#轉成對應關係
            melted_df.columns = ['from', 'to', 'Value']#欄位命名
            #melted_df = melted_df[melted_df['Value'] != 1].reset_index(drop=True)#刪除等於的1
            melted_df = melted_df[melted_df['Value'] > 0].reset_index(drop=True)#找大於0的值
            melted_df[['from', 'to']] = np.sort(melted_df[['from', 'to']], axis=1)
            melted_df = melted_df.drop_duplicates(subset=['from', 'to']).reset_index(drop=True)#刪除重複值
            
            #melted_df['drop_subset'] = melted_df['from'].astype(str) + "_" + melted_df['to'].astype(str)
            
            value_list = melted_df["Value"].tolist()
            percentile = np.percentile(value_list, (threshold*100))#根據value_list算出閥值
            
            #melted_df_thres = melted_df[melted_df['Value'] > threshold].reset_index(drop=True)
            melted_df_thres = melted_df[melted_df['Value'] >= percentile].reset_index(drop=True)#取符合threshold的值
            melted_df_thres["Value"] = np.sqrt(melted_df_thres['Value'])#取平方根的值
            
            melted_df_thres['from_name'] = melted_df_thres['from'].map(dict(zip(v_index, col_index)))#新增from_name的欄位，值為透過索引映射到對應值
            melted_df_thres['to_name'] = melted_df_thres['to'].map(dict(zip(v_index, col_index)))#
            melted_df_thres['id'] = melted_df_thres['from_name'].astype(str) + "_" + melted_df_thres['to_name'].astype(str)
            #melted_df_thres[['from_name', 'to_name']] = np.sort(melted_df_thres[['from_name', 'to_name']], axis=1)
            #melted_df_thres = melted_df_thres.drop_duplicates(subset=['from_name', 'to_name']).reset_index(drop=True)
            
            #melted_df_thres['Unit'] = str(Unit)
            #melted_df_thres['type'] = str(type)
            #melted_df_thres['total_nodes_num'] = str(total_nodes_num)
            #melted_df_thres['threshold'] = str(threshold)
            
            Min, Max = melted_df_thres['Value'].min(), melted_df_thres['Value'].max()#求出Value欄位的Max,Min值
            melted_df_thres['edge_width'] = melted_df_thres['Value'].apply(lambda x: ((x - Min) / (Max - Min)))#edge的寬度計算
                  
            nodes_list = melted_df_thres['from_name'].tolist() + melted_df_thres['to_name'].tolist()
            nodes_list = list(set(nodes_list))#刪除重複值

        
        elif type == 'co-occurrence':
            if Unit == "Document":
                input_data = CO_doc
                choose_data = CR_doc
                
            elif Unit == "Sentence":
                input_data = CO_sen
                choose_data = CR_sen
            
            if(input_filter != "不篩選"):
                input_filter_list = origin_key_dict_pd[origin_key_dict_pd['label'] == input_filter].index.tolist()
                #v = choose_data[Z].tolist()
                #v = list(set(v) & set(input_filter_list))
                v = [(index, choose_data.loc[index, Z]) for index in input_filter_list]
            else:
                v = choose_data[Z].tolist()
                v = list(enumerate(v))
                #v = list(set(v) & set(input_filter_list))
                
            #v = choose_data[Z].tolist()
            #v = list(enumerate(v))
            v = sorted(v, key=lambda x: x[1], reverse=True)
            v_index = [i for i, _ in v][:total_nodes_num]
            col_index = [((input_data.columns).tolist())[i] for i in v_index]
            x = input_data.loc[v_index, col_index]
            x.columns = v_index
            #x = (input_data.loc[v_index, col_index]).set_index(pd.Index(col_index))
            
            x_values = x.values# 獲取x的數據部分，轉換為numpy數組
            lower_triangle = np.tri(*x_values.shape, dtype=bool, k=0)# 獲取下三角部分的boolean
            x_values[lower_triangle] = 0# 將下三角部分（包括對角線）的元素設置為0
            x_updated = pd.DataFrame(x_values, index=x.index, columns=x.columns)# 將更新後的numpy數組重新轉換為DataFrame
            
            melted_df = x_updated.stack().reset_index()
            melted_df.columns = ['from', 'to', 'Value']
            #melted_df = melted_df[melted_df['Value'] != 1].reset_index(drop=True)
            melted_df = melted_df[melted_df['Value'] > 0].reset_index(drop=True)
            melted_df[['from', 'to']] = np.sort(melted_df[['from', 'to']], axis=1)
            melted_df = melted_df.drop_duplicates(subset=['from', 'to']).reset_index(drop=True)
            
            #melted_df['drop_subset'] = melted_df['from'].astype(str) + "_" + melted_df['to'].astype(str)
            
            value_list = melted_df["Value"].tolist()
            percentile = np.percentile(value_list, (threshold*100))#根據value_list算出閥值
            
            #melted_df_thres = melted_df[melted_df['Value'] > threshold].reset_index(drop=True)
            melted_df_thres = melted_df[melted_df['Value'] >= percentile].reset_index(drop=True)
            melted_df_thres["Value"] = np.sqrt(melted_df_thres['Value'])
            melted_df_thres['from_name'] = melted_df_thres['from'].map(dict(zip(v_index, col_index)))
            melted_df_thres['to_name'] = melted_df_thres['to'].map(dict(zip(v_index, col_index)))
            melted_df_thres['id'] = melted_df_thres['from_name'].astype(str) + "_" + melted_df_thres['to_name'].astype(str)
            #melted_df_thres[['from_name', 'to_name']] = np.sort(melted_df_thres[['from_name', 'to_name']], axis=1)
            #melted_df_thres = melted_df_thres.drop_duplicates(subset=['from_name', 'to_name']).reset_index(drop=True)
            
            #melted_df_thres['Unit'] = str(Unit)
            #melted_df_thres['type'] = str(type)
            #melted_df_thres['total_nodes_num'] = str(total_nodes_num)
            #melted_df_thres['threshold'] = str(threshold)
            
            Min, Max = melted_df_thres['Value'].min(), melted_df_thres['Value'].max()
            melted_df_thres['edge_width'] = melted_df_thres['Value'].apply(lambda x: ((x - Min) / (Max - Min)))
                  
            nodes_list = melted_df_thres['from_name'].tolist() + melted_df_thres['to_name'].tolist()
            nodes_list = list(set(nodes_list))

        nodes = [{'id': node, 
                  'label': node, 
                  'group': origin_key_dict_pd[origin_key_dict_pd['keywords'] == node]['label'].to_string().split()[1],
                  'title': node + "({})".format(origin_key_dict_pd[origin_key_dict_pd['keywords'] == node]['label'].to_string().split()[1])
                  #'color': color_dict.get((origin_key_dict_pd[origin_key_dict_pd["keywords"] == node])["label"].values[0])
                  }
                  for node in nodes_list]
                  #'color': color_list[keyword_class_list.index((origin_key_dict_pd[origin_key_dict_pd["keywords"] == node])["label"].values[0])]} for node in nodes_list]
                  #'color': color_list[(keyword_class_list.index(origin_key_dict_pd[origin_key_dict_pd['keywords'] == node]['label'].to_string().split()[1]))]} for node in nodes_list]
        # nodes = [{'id': node, 'label': node} for node in nodes_list]
        #length 篩選edge_df中item1、item2存在於nodes_list，type大於threshold
        edges = [{'id' : row['from_name']+'_'+row['to_name'], 
                  'from': row['from_name'], 'to': row['to_name'], 
                  'classes': type, #cor or coo
                  'weight': row['Value'], 
                  'width':row['edge_width']*6,
                 'title': row['from_name']+'_'+row['to_name']}
                  #'length':row['edge_width']*30}# 
                  #for idx, row in melted_df_thres[(melted_df_thres['from_name'].isin(nodes_list) & 
                                                                      #melted_df_thres['to_name'].isin(nodes_list))][melted_df_thres['Value'] > threshold].iterrows()]
            
                  for idx, row in melted_df_thres[(melted_df_thres['from_name'].isin(nodes_list) & 
                                                                      melted_df_thres['to_name'].isin(nodes_list))].iterrows()]
                                                                      

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
external_stylesheets = ['https://unpkg.com/antd@3.1.1/dist/antd.css',
                        'https://rawgit.com/jimmybow/CSS/master/visdcc/DataTable/Filter.css',
                        'https://cdnjs.cloudflare.com/ajax/libs/vis/4.20.1/vis.min.css',
                        ]

app = Dash(__name__, external_stylesheets = external_stylesheets)

#cyto.load_extra_layouts()

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
#app.title = 'My Title'
#app.layout = html.Div(children)
app.layout = html.Div([
    #html.H1('Hello Dash'),
    #html.Div(html.H2("This is my app")),
    #html.Div([
        #dbc.Row(dbc.Col(html.Div("A single column"))),
        #]),
# =============================================================================
#     html.Div(
#         className="app-header",
#         children=[
#             html.Div('Plotly Dash', className="app-header--title")
#         ]
#     ),
# =============================================================================
    html.Div([
        html.P("國防 太空文集 NER單中心網路分析", style={'font-size': '30px'}),
        html.H6('以特定主題為中心，從文集中選出相關性最高的關鍵詞，並對它們進行社會網絡分析',
                style={'color': 'Green'}),
        ## 切換句子或篇下拉式選單
# =============================================================================
#         dcc.Dropdown(
#             id='dropdown_choose_SenorDoc',
#             value= "Document",
#             clearable=False,
#             options=[
#                 {'label': method, 'value': method}
#                 for i, method in enumerate(Sen_Doc_list)
#             ]
#         ),
# =============================================================================
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
            options=[
                {'label': method, 'value': method}
                for i, method in enumerate(filter_class_list)
            ]
        ),
        html.H6('針對網路圖的節點可以進行篩選',
                style={'color': 'Green'}),
# =============================================================================
#         ## 連結強度計算方式
#         dcc.Dropdown(
#             id='dropdown_CRorCO',
#             value='correlation',
#             clearable=False,
#             options=[
#                 {'label': name.capitalize(), 'value': name}
#                 for name in ['correlation', 'co-occurrence']
#             ]
#         ),
# =============================================================================
        dbc.Label("設定網路節點數量", style={'font-size': '12px'}),
        dcc.Slider(
                id="total_nodes_num_slider", min=4, max=20,step=1,
                marks={i: str(i) for i in range(21)},
                value=8
                #id="total_nodes_num_slider", min=0, max=20,step=1,
                #marks={i: str(i) for i in range(21)},
                #value=7
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
# =============================================================================
#         dcc.Markdown('''
#                 ![Legend](https://i.ibb.co/z2JR9bS/Legend4.png)
#         '''),
# =============================================================================
        # html.Div(id='cytoscape-tapNodeData-output'),
        # html.Div(id='cytoscape-tapEdgeData-output'),
    ],style = {'height' : 850 ,'width': '20%', 'display': 'inline-block','backgroundColor':'rgb(232, 237, 248)'}),
    html.Div([
        dcc.Markdown('''
                ![Legend](https://i.ibb.co/s6RL68v/Legend0716.png)
                
        ''',
        style={'width': '60%', 'height': 20}
        ),
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
                    'nodes': {
                        #'x': [100, 200, 300],  # 指定節點的横向位置
                        #'y': [100, 200, 300], 
                        'size': 15  # 调整節點的大小
                        }
                }
            

                        ),
            dcc.Tooltip(id="graph-tooltip"),
            
                    
    ],style = {'height' : '100%' ,'width': '60%', 'display': 'inline-block'}),
    
    html.Div([
        #文本元件
        dcc.Textarea(
            id='textarea-example',
            #   value='paragraph',
            style={'width': '100%', 'height': 350},
            disabled = True,
        ),
        html.Div([
            visdcc.DataTable(
                id         = 'table' ,
                box_type   = 'radio',
                style={'width': '100%', 'height': 500},
                data       = table_data
            ),
        ])
        # html.Div(id='textarea-example-output', style={'whiteSpace': 'pre-line'})
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


@app.callback(
    Output("threshold_slide", 'min'),
    Output("threshold_slide", 'max'),
    # Output("threshold_slide", 'step'),
    Output("threshold_slide", 'marks'),
    Output("threshold_slide", 'value'),
    Input("RadioItems_CRorCO", 'value')
)
def update_elements(type):#當dropdown-update-layout下拉選單的值發生變化時，會觸發
    # if type == 'correlation':
    min=0
    max=1
    # step = None
    marks={i/10: str(i/10) for i in range(11)}
    value=0.5
        
    if type == 'co-occurrence':
        min=0
        max=1
        marks={i/10: str(i/10) for i in range(11)}
        value=0.5
        #max=200
        #marks={i: str(i) for i in range(0, 201, 50)}
        #value=100
    return min, max, marks, value


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
#data = ['3D printing']

def node_recation(Unit, data, type, total_nodes_num, threshold):
    
    k = data[0]#所點擊的node值
    v = XX_Sent[k]#取關鍵詞矩陣
    v = np.where(v == 1)[0]#矩陣中值為1的索引
    v = v.tolist()
    #index = raw_S.loc[v].drop("sen_list", axis=1)
    index = raw_S.loc[v]#透過索引取值
    merged_df = pd.merge(index, senlabel, on=['doc_id', 'sen_id'])#透過'doc_id', 'sen_id'進行合併
    merged_df = pd.merge(merged_df, X, on='doc_id', how='left')#透過'doc_id'進行合併
    merged_df = merged_df.drop_duplicates(subset=['doc_id', 'sen_id'], keep='first').reset_index(drop=True)
    #merged_df[['doc_id', 'sen_id']] = np.sort(merged_df[['doc_id', 'sen_id']], axis=1)
    #merged_df = merged_df.drop_duplicates(subset=['doc_id', 'sen_id']).reset_index(drop=True)
    merged_df['artDate'] = pd.to_datetime(merged_df['artDate']).dt.date
    merged_df = merged_df.sort_values(by='artDate', ascending=False).reset_index(drop=True)
    
    return merged_df, k
    
# In[]
#data = 'ABL_UK'
#from_token = "GPS"
#to_token = "Ligado"

def edge_recation(Unit, data, type, total_nodes_num, threshold):
    
# =============================================================================
#     from_token = ''
#     to_token = ''
#     for i, j in zip(coo_df['item1'], coo_df['item2']):#所點擊的edge值拆分成from_token和to_token
#         if data[0] == '{}_{}'.format(i, j):
#             from_token = i
#             to_token = j
#             break
# =============================================================================
    from_to_token =  data[0].split("_")    
    from_token = from_to_token[0] 
    to_token = from_to_token[1]
     
    #print(from_token + to_token)   
    #print('test')
    if Unit == "Sentence":
        #XX = XX_Sent
        
        token_df = XX_Sent[[from_token, to_token]]#透過from_token和to_token取關鍵詞矩陣
        token_df['total'] = token_df[from_token] + token_df[to_token]
        token_df = token_df[token_df['total'] == 2]#取等於的值
        
        index = raw_S.loc[token_df.index.tolist()]
        merged_df2 = pd.merge(index, senlabel, on=['doc_id', 'sen_id'])
        merged_df2 = pd.merge(merged_df2, X, on='doc_id', how='left')
        merged_df2 = merged_df2.drop_duplicates(subset=['doc_id', 'sen_id'], keep='first').reset_index(drop=True)
        
        merged_df2['artDate'] = pd.to_datetime(merged_df2['artDate']).dt.date
        merged_df2 = merged_df2.sort_values(by='artDate', ascending=False).reset_index(drop=True)
        
    else:
        
        #XX = XX_Doc
        
        token_df = XX_Sent[[from_token, to_token]]
        token_df['total'] = token_df[from_token] + token_df[to_token]
        token_df = token_df[token_df['total'] >= 1]
        
        index = raw_S.loc[token_df.index.tolist()]
        merged_df2 = pd.merge(index, senlabel, on=['doc_id', 'sen_id'])
        merged_df2 = pd.merge(merged_df2, X, on='doc_id', how='left')
        merged_df2 = merged_df2.drop_duplicates(subset=['doc_id', 'sen_id'], keep='first').reset_index(drop=True)
        
        merged_df2['artDate'] = pd.to_datetime(merged_df2['artDate']).dt.date
        merged_df2 = merged_df2.sort_values(by='artDate', ascending=False).reset_index(drop=True)
        
    return merged_df2, from_token


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
        print(selection)
        merged_df, token = node_recation(Unit, selection['nodes'], total_nodes_num, type, threshold)
        for i, j, k, l in zip(merged_df['artDate'], merged_df['doc_id'], merged_df['sen_list'], merged_df['artUrl']):
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
        print(selection)
        merged_df2, token = edge_recation(Unit, selection['edges'], total_nodes_num, type, threshold)
        for i, j, k, l in zip(merged_df2['artDate'], merged_df2['doc_id'], merged_df2['sen_list'], merged_df2['artUrl']):
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
            {'title': 'Recent:{}({})'.format(token,len(merged_df2)),
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

@app.callback(
    Output('textarea-example', 'value'),
    Input('table', 'box_selected_keys')
)
def myfun(box_selected_keys): 
    #print([box_selected_keys[0]])
    if box_selected_keys == None:
        return ''
    else: 
        return merged_df['artContent'][box_selected_keys[0]]
    

# =============================================================================
# @app.callback(
#     Output("graph-tooltip", "show"),
#     Output("graph-tooltip", "bbox"),
#     Output("graph-tooltip", "children"),
#     Input("net", "hoverData"),
# )
# def display_hover(hoverData):
#     if hoverData is None:
#         return False, no_update, no_update
#     
#     pt = hoverData["points"][0]
#     bbox = pt["bbox"]
#     num = pt["pointNumber"]
#     
#     print(num)
# # =============================================================================
# #     df_row = df.iloc[num]
# #     
# #     tooltip_content = [
# #     html.Div([
# #         #html.Img(src=img_src, style={"width": "100%"}),
# #         html.H2(f"{name}", style={"color": "darkblue", "overflow-wrap": "break-word"}),
# #         html.P(f"{form}"),
# #         html.P(f"{desc}"),
# #     ], style={'width': '200px', 'white-space': 'normal'})
# # ]
# # =============================================================================
#     
#     return True, bbox, #tooltip_content
# 
# =============================================================================


app.run_server(debug=True, use_reloader=False)  # Turn off reloader if inside Jupyter