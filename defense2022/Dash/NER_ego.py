import visdcc  # pip install visdcc
import pandas as pd
from dash import Dash, html
from dash import dcc, no_update
from dash.dependencies import Input, Output, State
import dash_cytoscape as cyto  # pip install dash-cytoscape
import numpy as np
import dash_bootstrap_components as dbc
import gc
# import dash_bootstrap_components as dbc
from colorama import init, Fore, Back, Style
import re
from dash import dash_table


import sys
sys.setrecursionlimit(1500)
# In[]
# # 字典
origin_key_dict_pd = pd.read_csv('./NER_old/entityDict.csv')
# 關鍵字類別
keyword_class_list = ["com", "rocket", "org", "satellite", "term", "loc"]
# 類別顏色
# colortext_list2 = [Fore.RED + Style.BRIGHT, Fore.GREEN, Fore.MAGENTA, Fore.MAGENTA + Style.BRIGHT, Fore.CYAN + Style.BRIGHT, Fore.GREEN+ Style.BRIGHT]
COLOUR = ['rgb(141, 211, 199)', 'rgb(247, 129, 191)', 'rgb(190, 186, 218)',
          'rgb(251, 128, 114)', 'rgb(146, 208, 80)', 'rgb(253, 180, 98)']
# global colortext_list
colortext_list = ["#8DD3C7", "#F781BF",
                  "#BEBADA", "#FB8072", "#92D050", "#FDB462"]
Sen_Doc_list = ["Sentence", "Document"]
# In[]
X = pd.read_csv('./NER_old/doc_raw_data.csv')
XX_Sent = pd.read_csv('./NER_old/SenDTM.csv')
XX_Doc = pd.read_csv('./NER_old/DocDTM.csv')
senlabel = pd.read_csv('./NER_old/sen_label_table.csv')
doclabel = pd.read_csv('./NER_old/doc_label_table.csv')
raw_S = pd.read_csv('./NER_old/sen_raw_data.csv')
CR_doc = pd.read_csv('./NER_old/DocCR.csv')
CR_sen = pd.read_csv('./NER_old/SenCR.csv')
CO_doc = pd.read_csv('./NER_old/DocCO.csv')
CO_sen = pd.read_csv('./NER_old/SenCO.csv')

# In[]
# 測試用
# Z = "Carnegie_Mellon_University"
# Z = "3D printing"  input_filter = ["com", "rocket", "loc"] input_filter = ["term"]
# Unit = "Sentence"  type = 'correlation' total_nodes_num = 8  threshold = 0.5 input_filter = "com"
# input_filter = "不篩選"
# In[]
# 計算edge寬度用


def calculate_edge_width(x, Min, Max):
    if Max - Min > 0:
        return (x - Min) / (Max - Min)
    else:
        return x

# 網路圖函數 Unit：計算單位(句、篇) Z：中心節點字 type：計算單位（CO或CR）total_nodes_num:網路圖節點數量 threshold:計算閥值 input_filter:篩選遮罩參數


def get_element_modify(Unit, Z, type, total_nodes_num, threshold, input_filter):

    node_size_list = []
    input_filter_list = []
    # print(input_filter)

    # 按照條件篩選資料
    if type == 'correlation':
        if Unit == "Document":
            input_data = CR_doc
        elif Unit == "Sentence":
            input_data = CR_sen

        # 判斷是否有網路篩選遮罩，取出資料裡符合Z(input_filter_list)的值和索引
        if isinstance(input_filter, list):
            # input_filter_list = [index for index, label in enumerate(origin_key_dict_pd['label']) if label not in input_filter]
            input_filter_list = [
                index for index, (label, keyword) in enumerate(zip(origin_key_dict_pd['label'], origin_key_dict_pd['keywords']))
                if keyword == Z or label not in input_filter
            ]
            v = [(index, input_data.loc[index, Z])
                 for index in input_filter_list]

        else:
            v = input_data[Z].tolist()
            v = list(enumerate(v))

        v = sorted(v, key=lambda x: x[1], reverse=True)  # 降序排列
        v_index = [i for i, _ in v][:total_nodes_num]  # 取出前K個索引
        col_index = [((input_data.columns).tolist())[i]
                     for i in v_index]  # 獲取對應的欄位名
        x = input_data.loc[v_index, col_index]  # v_index,col_index取值
        x.columns = v_index
        del v
        gc.collect()

        x_values = x.values  # 獲取x的數據部分，轉換為numpy數組
        # 獲取下三角部分的boolean *x_values.shape:使用x_values數組的形狀來確定矩陣的行數和列數 dtype:設定矩陣資料型態 k:True或False比例
        lower_triangle = np.tri(*x_values.shape, dtype=bool, k=0)
        x_values[lower_triangle] = 0  # 將下三角部分（True）的元素設置為0
        # 將更新後的numpy數組重新轉換為DataFrame
        x_updated = pd.DataFrame(x_values, index=x.index, columns=x.columns)
        del x
        gc.collect()

        melted_df = x_updated.stack().reset_index()  # 轉成對應關係
        melted_df.columns = ['from', 'to', 'Value']  # 欄位命名
        melted_df = melted_df[melted_df['Value']
                              > 0].reset_index(drop=True)  # 找大於0的值

        # 按['from', 'to']排序，刪除重複值
        melted_df[['from', 'to']] = np.sort(melted_df[['from', 'to']], axis=1)
        melted_df = melted_df.drop_duplicates(
            subset=['from', 'to']).reset_index(drop=True)

        # 閥值計算
        value_list = melted_df["Value"].tolist()
        percentile = np.percentile(value_list, (threshold*100))

        melted_df_thres = melted_df[melted_df['Value'] >= percentile].reset_index(
            drop=True)  # 符合threshold的value
        melted_df_thres["Value"] = np.sqrt(melted_df_thres['Value'])  # 取平方根值
        del melted_df
        gc.collect()

        # 新增['from_name','to_name','id']的欄位，透過索引映射到對應值
        melted_df_thres['from_name'] = melted_df_thres['from'].map(
            dict(zip(v_index, col_index)))
        melted_df_thres['to_name'] = melted_df_thres['to'].map(
            dict(zip(v_index, col_index)))
        melted_df_thres['id'] = melted_df_thres['from_name'].astype(
            str) + "_" + melted_df_thres['to_name'].astype(str)

        # edge的寬度計算
        Min, Max = melted_df_thres['Value'].min(
        ), melted_df_thres['Value'].max()
        melted_df_thres['edge_width'] = melted_df_thres['Value'].apply(
            lambda x: calculate_edge_width(x, Min, Max))

        nodes_list = melted_df_thres['from_name'].tolist(
        ) + melted_df_thres['to_name'].tolist()
        nodes_list = list(set(nodes_list))

        # 字典對應節點的freq值
        for node in nodes_list:
            node_size_list.append(int(
                origin_key_dict_pd[origin_key_dict_pd['keywords'] == node]['freq'].to_string().split()[1]))

        # 用以計算節點大小
        size_total = sum(node_size_list)

    # 按照條件篩選資料
    elif type == 'co-occurrence':
        if Unit == "Document":
            input_data = CO_doc
            choose_data = CR_doc
        elif Unit == "Sentence":
            input_data = CO_sen
            choose_data = CR_sen

        # 判斷是否有網路篩選遮罩，取出資料裡符合Z(input_filter_list)的值和索引
        if isinstance(input_filter, list):
            # input_filter_list = [index for index, label in enumerate(origin_key_dict_pd['label']) if label not in input_filter]
            input_filter_list = [
                index for index, (label, keyword) in enumerate(zip(origin_key_dict_pd['label'], origin_key_dict_pd['keywords']))
                if keyword == Z or label not in input_filter
            ]
            v = [(index, choose_data.loc[index, Z])
                 for index in input_filter_list]
        else:
            v = choose_data[Z].tolist()
            v = list(enumerate(v))

        v = sorted(v, key=lambda x: x[1], reverse=True)  # 降序排列
        v_index = [i for i, _ in v][:total_nodes_num]  # 取出前K個索引
        col_index = [((input_data.columns).tolist())[i]
                     for i in v_index]  # 獲取對應的欄位名
        x = input_data.loc[v_index, col_index]  # v_index,col_index取值
        x.columns = v_index
        del v
        gc.collect()

        x_values = x.values  # 獲取x的數據部分，轉換為numpy數組
        # 獲取下三角部分的boolean *x_values.shape:使用x_values數組的形狀來確定矩陣的行數和列數 dtype:設定矩陣資料型態 k:True或False比例
        lower_triangle = np.tri(*x_values.shape, dtype=bool, k=0)
        x_values[lower_triangle] = 0  # 將下三角部分（包括對角線）的元素設置為0
        # 將更新後的numpy數組重新轉換為DataFrame
        x_updated = pd.DataFrame(x_values, index=x.index, columns=x.columns)
        del x
        gc.collect()

        melted_df = x_updated.stack().reset_index()  # 轉成對應關係
        melted_df.columns = ['from', 'to', 'Value']  # 欄位命名
        melted_df = melted_df[melted_df['Value']
                              > 0].reset_index(drop=True)  # 找大於0的值

        # 按['from', 'to']排序，刪除重複值
        melted_df[['from', 'to']] = np.sort(melted_df[['from', 'to']], axis=1)
        melted_df = melted_df.drop_duplicates(
            subset=['from', 'to']).reset_index(drop=True)

        # 閥值計算
        value_list = melted_df["Value"].tolist()
        percentile = np.percentile(
            value_list, (threshold*100))  # 根據value_list算出閥值

        melted_df_thres = melted_df[melted_df['Value'] >= percentile].reset_index(
            drop=True)  # 取符合threshold的value
        melted_df_thres["Value"] = np.sqrt(melted_df_thres['Value'])  # 取平方根值
        del melted_df
        gc.collect()

        # 新增['from_name','to_name','id']的欄位，值為透過索引映射到對應值
        melted_df_thres['from_name'] = melted_df_thres['from'].map(
            dict(zip(v_index, col_index)))
        melted_df_thres['to_name'] = melted_df_thres['to'].map(
            dict(zip(v_index, col_index)))
        melted_df_thres['id'] = melted_df_thres['from_name'].astype(
            str) + "_" + melted_df_thres['to_name'].astype(str)

        # edge的寬度計算
        Min, Max = melted_df_thres['Value'].min(
        ), melted_df_thres['Value'].max()
        melted_df_thres['edge_width'] = melted_df_thres['Value'].apply(
            lambda x: calculate_edge_width(x, Min, Max))

        nodes_list = melted_df_thres['from_name'].tolist(
        ) + melted_df_thres['to_name'].tolist()
        nodes_list = list(set(nodes_list))  # 刪除重複值

        # 字典對應節點的freq值
        for node in nodes_list:
            node_size_list.append(int(
                origin_key_dict_pd[origin_key_dict_pd['keywords'] == node]['freq'].to_string().split()[1]))

        # 用以計算節點大小
        size_total = sum(node_size_list)

    # group:節點字類別 title:網路圖tooltip shape:節點形狀 size:節點大小
    nodes = [
        {
            'id': node,
            'label': node,
            'group': origin_key_dict_pd[origin_key_dict_pd['keywords'] == node]['label'].to_string().split()[1],
            'title': node + ":({},{})".format(origin_key_dict_pd[origin_key_dict_pd['keywords'] == node]['label'].to_string().split()[1], origin_key_dict_pd[origin_key_dict_pd['keywords'] == node]['freq'].to_string().split()[1]),
            'shape': 'dot',
            'size': 15 + (int(origin_key_dict_pd[origin_key_dict_pd['keywords'] == node]['freq'].to_string().split()[1])/size_total)*100,
        }
        for node in nodes_list
    ]

    # width:邊寬度 title:網路圖tooltip
    edges = [
        {
            'id': row['from_name']+'_'+row['to_name'],
            'from': row['from_name'],
            'to': row['to_name'],
            'classes': type,  # cor or coo
            'weight': row['Value'],
            'width':row['edge_width']*6,
            'title': row['from_name']+'_'+row['to_name']
        }
        for idx, row in melted_df_thres[(melted_df_thres['from_name'].isin(nodes_list) &
                                         melted_df_thres['to_name'].isin(nodes_list))].iterrows()
    ]

    info = {"Unit": str(Unit),
            "type": str(type),
            "total_nodes_num": total_nodes_num,
            "threshold": threshold
            }

    data = {'nodes': nodes,
            'edges': edges,
            'info': info
            }

    return data
# In[]


# 外部css元件
external_stylesheets = ['https://unpkg.com/antd@3.1.1/dist/antd.css',
                        'https://rawgit.com/jimmybow/CSS/master/visdcc/DataTable/Filter.css',
                        'https://cdnjs.cloudflare.com/ajax/libs/vis/4.20.1/vis.min.css',
                        ]

app = Dash(__name__, external_stylesheets=external_stylesheets)


styles = {
    'pre': {
        'border': 'thin lightgrey solid',  # 邊框屬性
        'overflowX': 'scroll'  # 內容水平延伸
    }
}

# dash_table_columns = []
# table_data = pd.DataFrame(columns=['Version1', 'Version2', 'Version3'])
merged_df = pd.DataFrame()
# set my legend
propotion = 100/len(COLOUR)
legend = []
for c, label in zip(COLOUR, keyword_class_list):
    l = html.Div(label,
                 style={
                     'background-color': c,
                     'padding': '20px',
                     'color': 'white',
                     'display': 'inline-block',
                     'width': str(propotion)+'%',
                     'font-size': '20px'
                 })
    legend.append(l)

bold_orange = {
    'font-size': '16px',
    'color': '#CA774B',
    'font-weight': 'bold',
    'display': 'block',
    'margin': '1rem 0rem 0rem 0rem'}  # top,right,bottom,left

inline_orange = {
    'font-size': '16px',
    'color': '#CA774B',
    'font-weight': 'bold',
    'display': 'inline-block',
    'margin': '0.5rem 1.5rem 0rem 0rem'}

annotation = {'font-size': '14px', 'color': '#66828E'}
# global res
table_data = {
    'dataSource': [],
    'columns': [{'title': 'Date',
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
                ]
}

app.layout = html.Div(children=[
    html.H1("國防太空文集 NER單中心網路分析",
            style={
                'font-size': '36px',
                'textAlign': 'center',
                'backgroundColor': '#daf5ed',
                'margin': '0px',
                'font-weight': 'bold',
                'padding': '5px'
            }
            ),
    html.H6('以特定主題為中心，從文集中選出相關性最高的關鍵詞，並對它們進行社會網絡分析',
            style={
                'font-size': '24px',
                'textAlign': 'center',
                'backgroundColor': '#f2efe4',
                'padding': '3px',
                'margin': '0px',
            }
            ),
    html.Div([
        html.Div([
            dbc.Label("選擇關鍵字類別",
                      style=bold_orange),
            # 切換類別下拉式選單
            dcc.Dropdown(
                id='dropdown_choose_class',
                value=4,
                clearable=False,
                options=[
                    {'label': clas, 'value': i}
                    for i, clas in enumerate(keyword_class_list)
                ],
                style={'margin': '0.5rem 0rem 0.8rem 0rem'}
            ),
            dbc.Label("選擇關鍵字",
                      style=bold_orange),
            # 選擇中心詞下拉式選單
            dcc.Dropdown(
                id='dropdown_choose_name',
                value='3D printing',
                clearable=False,
                options=[
                    {'label': name, 'value': name}
                    for name in origin_key_dict_pd[origin_key_dict_pd['label'] == keyword_class_list[0]]['keywords'].to_list()
                ],
                style={'margin': '0.5rem 0rem 0.8rem 0rem'}
            ),
            dbc.Label("網路篩選遮罩",
                      style=bold_orange),
            # 網路篩選遮罩下拉式選單
            dcc.Dropdown(
                id='dropdown_choose_filter',
                clearable=False,
                multi=True,
                options=[
                    {'label': method, 'value': method}
                    for i, method in enumerate(keyword_class_list)
                ],
                style={'margin': '0.5rem 0rem 0rem 0rem'}
            ),
            html.H6('針對網路圖的節點類別進行篩選',
                    style=annotation),

            dbc.Label("設定網路節點數量",
                      style=inline_orange),
            dcc.Dropdown(
                id='total_nodes_num',
                options=[{'label': str(i), 'value': i}
                         for i in range(3, 10)],
                value=8,
                style={
                    'verticalAlign': 'top',
                    'margin': '0rem 1.5rem 0rem 0rem',
                    'display': 'inline-block'
                }
            ),
            dbc.Label("依關聯節度篩選鏈結",
                      style=bold_orange),
            # 網路圖篩選節點閥值slider
            dcc.Slider(
                id="threshold_slide", min=0, max=1, step=0.01,
                tooltip={
                    "placement": "bottom",
                    "always_visible": True,
                },
                marks={i/10: str(i/10) for i in range(51)},
                value=0.5
            ),
            dbc.Label("字詞連結段落", style=inline_orange),
            # 計算單位選鈕
            dcc.RadioItems(
                id='RadioItems_SenorDoc',
                options=[{'label': '句 ', 'value': 'Sentence'},
                         {'label': '篇', 'value': 'Document'},],
                value='Sentence',
                inline=True,
                style={'margin': '0.5rem 1rem 0rem 0rem',
                       'display': 'inline-block'}
            ),
            dbc.Label("連結強度計算方式",
                      style=bold_orange),
            dcc.RadioItems(
                id='RadioItems_CRorCO',
                options=[{'label': '共同出現次數', 'value': 'co-occurrence'},
                         {'label': '相關係數', 'value': 'correlation'},],
                value='correlation',
                inline=True,
                style={'margin': '0.5rem 0rem 0rem 0rem'}
            ),
            dbc.Label("連結強度依據字詞出現頻率", style=annotation),
            html.Br(),
            dbc.Label("較高，可選「相關係數」", style=annotation),
            html.Br(),
            dbc.Label("較低，可擇「共同出現次數」", style=annotation),
        ],
            style={
            'background-color': '#daf5ed',
            'display': 'inline-block',
            'width': '15%',
            'height': '900px',
            'padding': '0.5%'}
        ),
        html.Div([
            # legend
            html.Div(legend,
                     style={
                         'background-color': "#ede7d1",
                         'color': '#f2efe4',
                         'height': '7.5%',
                         'text-align': 'center',
                         'font-size': '24px',
                         'padding': '0px'}),
            # 網路圖
            visdcc.Network(
                id='net',
                selection={'nodes': [], 'edges': []},
                options={
                    'interaction': {
                        'hover': True,
                        'tooltipDelay': 300,
                    },
                    'groups': {
                        keyword_class_list[0]: {'color': COLOUR[0]},
                        keyword_class_list[1]: {'color': COLOUR[1]},
                        keyword_class_list[2]: {'color': COLOUR[2]},
                        keyword_class_list[3]: {'color': COLOUR[3]},
                        keyword_class_list[4]: {'color': COLOUR[4]},
                        keyword_class_list[5]: {'color': COLOUR[5]},

                    },
                    'autoResize': True,
                    'height': '800px',
                    'width': '100%',
                    'layout': {
                        'improvedLayout': True,
                        'hierarchical': {
                            'enabled': False,
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
                    'physics': {
                        'enabled': True,
                        'barnesHut': {
                            'theta': 0.5,
                            'gravitationalConstant': -20000,  # repulsion強度
                            'centralGravity': 0.3,
                            'springLength': 95,
                            'springConstant': 0.04,
                            'damping': 0.09,
                            # 'avoidOverlap': 0.01
                        },
                    },
                    'adaptiveTimestep': True,
                }
            ),
        ], style={'display': 'inline-block',
                  'width': '50%',
                  'verticalAlign': 'top'}
        ),
        # 放置文章
        html.Div([
            # 文本元件
            dcc.Textarea(
                id='textarea-example',
                #   value='paragraph',
                style={'width': '100%', 'height': '480px'},
                disabled=True,
            ),
            # =============================================================================
            #             dash_table.DataTable(
            #                 id         = 'table' ,
            #                 css=[dict(selector="p", rule="margin: 0px; text-align: center")],
            #                 style_cell={"textAlign": "center",'color': "#000000"},
            #                 data = table_data.to_dict('records'),
            #                 #data = table_data,
            #                 #data = table_data['dataSource'],
            #                 #data = res,
            #                 #columns = [{"name": i, "id": i} for i in table_data.columns],
            #                 #rows = table_data,
            #                 columns = [
            #                     {"name": "Version1", "id": "Version1"},
            #                     {"name": "Version2", "id": "Version2"},
            #                     {"name": "Version3", "id": "Version3"},
            #                     ],
            #                 markdown_options={"html": True},
            #                 ),
            # =============================================================================
            visdcc.DataTable(
                id='table',
                box_type='radio',
                style={'width': '100%',
                       'height': '100%'},
                data=table_data,
                pagination={'pageSize': 10},
            ),
        ], style={
            'background-color': COLOUR[0],
            'color': 'white',
            'display': 'inline-block',
            'width': '35%',
            'height': '150%',
            'verticalAlign': 'top'}),
    ], style={'height': '100%', 'width': '100%'}),

])


# 切換 class 下拉式選單
@app.callback(
    Output("dropdown_choose_name", 'options'),
    Input("dropdown_choose_class", "value"),
)
def update_elements(class_idx):  # 當dropdown_choose_class下拉選單的值發生變化時，會觸發，class_idx類別索引
    # 選擇中心詞
    options = [
        {'label': name, 'value': name}
        for name in origin_key_dict_pd[origin_key_dict_pd['label'] == keyword_class_list[class_idx]]['keywords'].to_list()
    ]

    return options


# 更新下拉選單
@app.callback(
    Output("threshold_slide", 'min'),
    Output("threshold_slide", 'max'),
    Output("threshold_slide", 'marks'),
    Output("threshold_slide", 'value'),
    Input("RadioItems_CRorCO", 'value')
)
def update_elements(type):
    # if type == 'correlation':
    min = 0
    max = 1
    marks = {i/10: str(i/10) for i in range(11)}
    value = 0.3

    if type == 'co-occurrence':
        min = 0
        max = 1
        marks = {i/10: str(i/10) for i in range(11)}
        value = 0.3

    return min, max, marks, value

# 當dropdown-update-layout下拉選單的值發生變化時，更新網路圖


@app.callback(
    Output("net", 'data'),
    Input('RadioItems_SenorDoc', 'value'),
    Input("dropdown_choose_name", 'value'),
    Input("total_nodes_num", "value"),
    Input('RadioItems_CRorCO', 'value'),
    Input('threshold_slide', 'value'),
    Input('dropdown_choose_filter', 'value'),
)
def update_elements(Unit, center_node, total_nodes_num, type, threshold, input_filter):

    return get_element_modify(Unit, center_node, type, total_nodes_num, threshold, input_filter)

# In[]
# 測試用
# data = ['NASA']
# data = ['3D printing']
# In[]


def node_recation(Unit, data, type, total_nodes_num, threshold):

    colored_sen_list = []

    k = data[0]  # 所點擊的node值
    v = XX_Sent[k]  # 取關鍵詞矩陣
    v = np.where(v == 1)[0]  # 矩陣中值為1的索引
    v = v.tolist()
    index = raw_S.loc[v]  # 透過索引取值

    # 資料合併
    merged_df = pd.merge(index, senlabel, on=['doc_id', 'sen_id'])
    merged_df = pd.merge(merged_df, X, on='doc_id', how='left')
    merged_df = merged_df.drop_duplicates(
        subset=['doc_id', 'sen_id'], keep='first').reset_index(drop=True)

    # 資料按時間排序
    merged_df['date'] = pd.to_datetime(merged_df['date']).dt.date
    merged_df = merged_df.sort_values(
        by='date', ascending=False).reset_index(drop=True)

    if len(merged_df) > 1000:
        merged_df = merged_df[:999]

    for index, row in merged_df.iterrows():

        label = row["label"]
        start = int(row["start"])
        end = int(row["end"])
        ner_sen = row["ner_sen"]

        list_index = (keyword_class_list.index(label))
        # text_color = colortext_list[list_index]
        text_color = colortext_list[list_index]
        # target_colored_text = ner_sen[start:end]
        # colored_text = text_color + target_colored_text + Style.RESET_ALL
        # colored_text = ner_sen.replace(target_colored_text, colored_text)
        # colored_text = html.Div()
        # span_tag = f"<span style='color: {text_color};'>"

# =============================================================================
#         colored_text = re.sub(
#             re.escape(ner_sen[start:end]),  # 將區間的文字進行正確的正則表達式轉義
#             span_tag + "\\g<0></span>",  # 在找到的文本周圍包裹span標籤
#             ner_sen
#             )
# =============================================================================
        colored_text = "{}<span style='color: {};'>{}</span>{}".format(
            ner_sen[:start], text_color, ner_sen[start:end], ner_sen[end:])

        colored_sen_list.append(colored_text)

    merged_df["colored_sen"] = colored_sen_list

    # text_color = colortext_list[(keyword_class_list.index(row["label"]))]

    # merged_df['artDate_Url'] = merged_df.apply(lambda row: f'<a href="{row["artUrl"]}">{row["artDate"]}</a>', axis=1)
    merged_df['artDate_Url'] = merged_df.apply(
        lambda row: html.A(html.P(row['date']), href=row['link']), axis=1)

    return merged_df, k

# In[]
# 測試用
# data = ['ABL_UK']
# from_token = "3D printing"
# to_token = "ArianeGroup"
# In[]


def edge_recation(Unit, data, type, total_nodes_num, threshold):

    colored_sen_list = []
    # from,to token
    from_to_token = data[0].split("_")
    from_token = from_to_token[0]
    to_token = from_to_token[1]

    if Unit == "Sentence":
        token_df = XX_Sent[[from_token, to_token]]  # from_token,to_token取關鍵詞矩陣

        token_df['total'] = token_df[from_token] + token_df[to_token]
        token_df = token_df[(token_df[from_token] == 1)
                            & (token_df[to_token] == 1)]
        index = raw_S.loc[token_df.index.tolist()]  # index取值

        # 欄位合併，刪除重複值
        merged_df2 = pd.merge(index, senlabel, on=['doc_id', 'sen_id'])
        merged_df2 = pd.merge(merged_df2, X, on='doc_id', how='left')
        merged_df2 = merged_df2.drop_duplicates(
            subset=['doc_id', 'sen_id'], keep='first').reset_index(drop=True)

        # 資料型態轉為datatime後，降序排列
        merged_df2['date'] = pd.to_datetime(merged_df2['date']).dt.date
        merged_df2 = merged_df2.sort_values(
            by='date', ascending=False).reset_index(drop=True)

    else:
        token_df = XX_Sent[[from_token, to_token]]  # from_token,to_token取關鍵詞矩陣

        token_df['total'] = token_df[from_token] + token_df[to_token]
        token_df = token_df[token_df['total'] >= 1]
        index = raw_S.loc[token_df.index.tolist()]  # index取值

        # 欄位合併，刪除重複值
        merged_df2 = pd.merge(index, senlabel, on=['doc_id', 'sen_id'])
        merged_df2 = pd.merge(merged_df2, X, on='doc_id', how='left')
        merged_df2 = merged_df2.drop_duplicates(
            subset=['doc_id', 'sen_id'], keep='first').reset_index(drop=True)

        # 資料型態轉為datatime後，降序排列
        merged_df2['date'] = pd.to_datetime(merged_df2['date']).dt.date
        merged_df2 = merged_df2.sort_values(
            by='date', ascending=False).reset_index(drop=True)

    # 防止datatable元件過載
    if len(merged_df2) > 1000:
        merged_df2 = merged_df2[:999]

    for index, row in merged_df2.iterrows():

        label = row["label"]
        start = int(row["start"])
        end = int(row["end"])
        ner_sen = row["ner_sen"]

        list_index = (keyword_class_list.index(label))
        # text_color = colortext_list[list_index]
        text_color = colortext_list[list_index]
        target_colored_text = ner_sen[start:end]
        colored_text = text_color + target_colored_text + Style.RESET_ALL
        colored_text = ner_sen.replace(target_colored_text, colored_text)
        # colored_text = html.Div()
        colored_sen_list.append(colored_text)

    merged_df2["colored_sen"] = colored_sen_list

    # 將artDate加入超連結功能，測試中
    merged_df2['artDate_Url'] = merged_df2.apply(
        lambda row: html.A(html.P(row['date']), href=row['link']), axis=1)

    return merged_df2, from_token, to_token


# Datatable更新函數
@app.callback(
    Output('table', 'data'),
    # Output('table1','children'),
    Input('RadioItems_SenorDoc', 'value'),
    Input('net', 'selection'),
    Input("total_nodes_num", "value"),
    Input('RadioItems_CRorCO', 'value'),
    Input('threshold_slide', 'value'),
)
def update_elements(Unit, selection, total_nodes_num, type, threshold):
    global merged_df
    res = []

    if len(selection['nodes']) != 0:
        # print(selection)
        # 將node對應資料映射到datatable
        merged_df, token = node_recation(
            Unit, selection['nodes'], total_nodes_num, type, threshold)
        for i, j, k, l in zip(merged_df['date'], merged_df['doc_id'], merged_df['colored_sen'], merged_df['link']):
            res.append({'Date': i, 'id': j, 'Recent': k, 'url': l})
        table_data['columns'] = [
            {'title': 'Date',
             'dataIndex': 'Date',
             'key': 'Date',
             'width': '25%'},
            {'title': 'doc_id',
             'dataIndex': 'id',
             'key': 'id',
             'width': '15%'},
            {'title': 'Recent:{}({})'.format(token, len(merged_df)),
             'dataIndex': 'Recent',
             'key': 'Recent',
             'width': '60%'},
        ]
    elif len(selection['edges']) != 0:
        # print(selection)
        # 將edge對應資料映射到datatable
        merged_df2, from_token, to_token = edge_recation(
            Unit, selection['edges'], total_nodes_num, type, threshold)
        for i, j, k, l in zip(merged_df2['date'], merged_df2['doc_id'], merged_df2['colored_sen'], merged_df2['link']):
            res.append({'Date': i, 'id': j, 'Recent': k, 'url': l})
        table_data['columns'] = [
            {'title': 'Date',
             'dataIndex': 'Date',
             'key': 'Date',
             'width': '25%'},
            {'title': 'doc_id',
             'dataIndex': 'id',
             'key': 'id',
             'width': '15%'},
            {'title': 'Recent:{}({})'.format(from_token + "_" + to_token, len(merged_df2)),
             'dataIndex': 'Recent',
             'key': 'Recent',
             'width': '60%'},
        ]
    else:
        table_data['columns'] = [
            {'title': 'Date',
             'dataIndex': 'Date',
             'key': 'Date',
             'width': '25%'},
            {'title': 'doc_id',
             'dataIndex': 'id',
             'key': 'id',
             'width': '15%'},
            {'title': 'Recent',
             'dataIndex': 'Recent',
             'key': 'Recent',
             'width': '60%'},
            # {'title': 'url',
            # 'dataIndex': 'url',
            # 'key': 'url',
            # 'width': '15%'}
        ]

    table_data['dataSource'] = res

    return table_data
# =============================================================================
#     global merged_df
#     global table_data
#     global dash_table_columns
#
#     res = []
#
#     if len(selection['nodes']) != 0:
#         print(selection)
#         print((selection['nodes']))
#         #將node對應資料映射到datatable
#         merged_df, token = node_recation(Unit, selection['nodes'], total_nodes_num, type, threshold)
#         #table_data = merged_df
#         table_data = pd.DataFrame(columns=['Version1', 'Version2', 'Version3'])
#
#         #for i, j, k, l in zip(merged_df['date'], merged_df['doc_id'], merged_df['colored_sen'], merged_df['link']):
#            # res.append({'Date':i, 'id':j, 'Recent':k, 'url':l})
#         #for i, j, k in zip(merged_df['date'], merged_df['doc_id'], merged_df['colored_sen']):
#             #table_data = table_data.append({'Date': i, 'id': j, 'Recent': k}, ignore_index=True)
#             #pd.concat(table_data,pd.DataFrame({'Version1': i,
#                                #'Version2': j,
#                                #'Version3': k,
#                                #}))
#         table_data['Date'] = merged_df['date'].tolist()
#         table_data['id'] = merged_df['doc_id'].tolist()
#         table_data['Recent'] = merged_df['colored_sen'].tolist()
#
# # =============================================================================
# #         table_data['columns'] = [
# #             {'title': 'Date',
# #             'dataIndex': 'Date',
# #             'key': 'Date',
# #             'width': '20%'},
# #             {'title': 'doc_id',
# #             'dataIndex': 'id',
# #             'key': 'id',
# #             'width': '20%'},
# #             {'title': 'Recent:{}({})'.format(token,len(merged_df)),
# #             'dataIndex': 'Recent',
# #             'key': 'Recent',
# #             'width': '60%'},
# #             #{'title': 'url',
# #             #'dataIndex': 'url',
# #             #'key': 'url',
# #             #'width': '15%'}
# #         ]
# # =============================================================================
#     elif len(selection['edges']) != 0:
#         #print(selection)
#         #將edge對應資料映射到datatable
#         merged_df2, from_token, to_token = edge_recation(Unit, selection['edges'], total_nodes_num, type, threshold)
#         #table_data = merged_df2
#         for i, j, k, l in zip(merged_df2['date'], merged_df2['doc_id'], merged_df2['colored_sen'], merged_df2['link']):
#             res.append({'Date':i, 'id':j, 'Recent':k, 'url':l})
#         table_data['columns'] = [
#             {'title': 'Date',
#             'dataIndex': 'Date',
#             'key': 'Date',
#             'width': '20%'},
#             {'title': 'doc_id',
#             'dataIndex': 'id',
#             'key': 'id',
#             'width': '20%'},
#             {'title': 'Recent:{}({})'.format(from_token + "_" + to_token,len(merged_df2)),
#             'dataIndex': 'Recent',
#             'key': 'Recent',
#             'width': '60%'},
#             #{'title': 'url',
#             #'dataIndex': 'url',
#             #'key': 'url',
#             #'width': '15%'}
#         ]
# # =============================================================================
# #     else:
# #         table_data['columns'] = [
# #             {'title': 'Date',
# #             'dataIndex': 'Date',
# #             'key': 'Date',
# #             'width': '20%'},
# #             {'title': 'doc_id',
# #             'dataIndex': 'id',
# #             'key': 'id',
# #             'width': '20%'},
# #             {'title': 'Recent',
# #             'dataIndex': 'Recent',
# #             'key': 'Recent',
# #             'width': '60%'},
# #             #{'title': 'url',
# #             #'dataIndex': 'url',
# #             #'key': 'url',
# #             #'width': '15%'}
# #         ]
# # =============================================================================
# # =============================================================================
# #         dash_table_columns = [
# #             {"name": "Version", "id": "Version"},
# #             {"name": "Version", "id": "Version"},
# #             {"name": "Version", "id": "Version"},
# #             ],
# # =============================================================================
#
#         #data = table_data
#     #table_data['dataSource'] = res
#     #table_data = res
#     #table_data = table_data.to_dict('records')
#     #print(table_data)
#     #print(table_data.to_dict('records'))
#     #da = table_data.to_dict('records')
#     #print(table_data)
#     #return table_data
#     #return res
#     #return table_data.to_dict('records')
#     #return dash_table.DataTable(data = table_data.to_dict('records') , columns = dash_table_columns)
#     #return table_data
# =============================================================================

# textarea更新函數


@app.callback(
    Output('textarea-example', 'value'),
    Input('table', 'box_selected_keys')
)
def myfun(box_selected_keys):
    # print(box_selected_keys)
    # print(merged_df['doc_id'][box_selected_keys[0]])
    if box_selected_keys == None:
        return ''
    else:
        # sentence_list[word_start] = f'<span><font color="{label_color}"><b>{sentence_list[word_start]}</b></font>'
        # sentence_list[word_end] = f'{sentence_list[word_end]}</b></font><sub style="color:{label_color};">{label_text}</sub></span>'
        color_doc_id = (merged_df['doc_id'][box_selected_keys[0]])
        color_doc = (merged_df['ner_doc'][box_selected_keys[0]])

        color_doc_id_df = doclabel[doclabel["doc_id"] == color_doc_id]
        color_doc_id_df = color_doc_id_df.drop_duplicates(
            subset=['doc_kw_list']).reset_index(drop=True)

        for index, row in color_doc_id_df.iterrows():

            label = row["label"]
            # start = int(row["start"])
            # end = int(row["end"])
            doc_kw_list = row["doc_kw_list"]

            list_index = (keyword_class_list.index(label))
            text_color = colortext_list[list_index]
            colored_word = f'<span style="color:{text_color}">{doc_kw_list}</span>'

            color_doc = color_doc.replace(doc_kw_list, colored_word)

        color_doc = html.Div(color_doc)

        return merged_df['ner_doc'][box_selected_keys[0]]
        # return color_doc


# Turn off reloader if inside Jupyter
app.run_server(debug=True, use_reloader=False)
