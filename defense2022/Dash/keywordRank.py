from dash import Dash, html, dcc, callback, Output, Input
import pandas as pd
import plotly.graph_objects as go

dtm = pd.read_csv('freq_DTM.csv')

# segment labels by colours
# capital setting
COULOUR = ["#CA774B", "#CC5F5A", "#66828E", "#FEC37D",
           "#678F74", "#D4C3AA"]
E_CLASS = ["term", "loc", "com", "rocket", "satellite", "org"]
dtm.insert(2, 'colour', dtm['label'])
for i in range(0, len(E_CLASS)):
    dtm['colour'] = dtm['colour'].replace({E_CLASS[i]: COULOUR[i]})

propotion = 100/len(COULOUR)
# make a decoration under my title
dacoration = []
# set my legend
legend = []
for c, label in zip(COULOUR, E_CLASS):
    d = html.Div(" ", style={
                 'background-color': c,
                 'padding': '5px',
                 'color': 'white',
                 'display': 'inline-block',
                 'width': str(propotion-1.5)+'%'})
    dacoration.append(d)
    l = html.Div(label,
                 style={
                     'background-color': c,
                     'padding': '20px',
                     'color': 'white',
                     'display': 'inline-block',
                     'width': str(propotion*0.75)+'%',
                     'font-size': '26px'  # Adjust the font size here
                 })
    legend.append(l)

app = Dash(__name__)
app.layout = html.Div([
    html.H1(children='國防 SpaceNews 關鍵字詞頻率排名', style={'textAlign': 'center'}),
    html.Div(dacoration, style={'text-align': 'center'}),
    html.H3(children='選擇關鍵字排名區間', style={'textAlign': 'left'}),
    dcc.RangeSlider(min=1, max=50, step=1, value=[1, 5], tooltip={
        "placement": "bottom", "always_visible": True}, id='K'),
    html.Div(legend, style={'text-align': 'center', }),
    dcc.Loading(dcc.Graph(id="graph"), type="cube",),
])


@app.callback(
    Output('graph', 'figure'),
    Input('K', 'value'),
)
def display_animated_graph(K):

    # get latest 36 months
    months = dtm.columns.values.flatten().tolist()[-36:]
    # set sliders_dict
    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": "Year-Month:",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 300, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": []
    }
    # add step in sliders_dict
    sliders_dict['steps'] = [
        {
            # the dynamic frame is changed depend on "args", a list containing the month to be shown
            "args": [[month], {"frame": {"duration": 300, "redraw": False}, "mode": "immediate"}],
            # "label" is what the user sees in the slider to select a specific frame
            "label": month,
            "method": "animate"
        }
        for month in months
    ]

    # make the dict fn
    fn = {}
    for m in months:
        # use nlargest to order keywords by frequency(counts) per month
        # K[1] is the last keyword rank u want to get
        df = dtm.nlargest(n=K[1], columns=[m])[['keywords', 'colour', m]]
        # get the specific range of rank
        df = df[K[0]-1:]
        # Melt the DataFrame to reshape it
        df_melted = pd.melt(
            df, id_vars=['keywords', 'colour'], var_name='months', value_name='count')
        # Sort the DataFrame by keywords and months
        df_melted = df_melted.sort_values(by=['months', 'count'])
        # set key as month
        fn[m] = df_melted

    fig1 = go.Figure(
        data=[
            go.Bar(
                x=fn[months[0]]['count'],
                y=fn[months[0]]['keywords'],
                orientation='h',
                hoverinfo='all',
                textposition='auto',  # position text outside bar
                insidetextanchor='start',
                text=fn[months[0]]['keywords'],
                texttemplate='%{y} %{x}',
                textfont={'size': 18},
                width=0.9, marker={'color': fn[months[0]]['colour']})
        ],
        layout=go.Layout(
            xaxis=dict(range=[0, fn[months[0]]['count'].max()*1.42+100], autorange=False,
                       title=dict(text='count',
                                  font=dict(size=18))),
            yaxis=dict(range=[-1, (K[1]-K[0]+1)],
                       autorange=False,
                       showline=False,
                       visible=True,  # to show the title
                       showticklabels=False,  # to avoid displaying the names
                       title=dict(
                           text='keywords', font=dict(size=18)),
                       tickfont=dict(size=14)),
            title=dict(
                x=0.5, xanchor='center',
                # Use the first key from fn,
                text=f'Amounts of top ' + \
                str(K[0]) + ' to '+str(K[1]) + ' keywords appear per month',
                font=dict(size=28)),

            # Add button
            updatemenus=[
                {
                    "buttons": [
                        {
                            "args": [None, {"frame": {"duration": 800, "redraw": False},
                                            "fromcurrent": True, "transition": {"duration": 300,
                                                                                "easing": "quadratic-in-out"}}],
                            "label": "Play",
                            "method": "animate"
                        },
                        {
                            "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                              "mode": "immediate",
                                              "transition": {"duration": 0}}],
                            "label": "Pause",
                            "method": "animate"
                        }
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 87},
                    "showactive": False,
                    "type": "buttons",
                    "x": 0.1,
                    "xanchor": "right",
                    "y": 0,
                    "yanchor": "top"
                }

            ],
            sliders=[sliders_dict]
        ),
        frames=[
            go.Frame(
                data=[
                    go.Bar(
                        orientation='h',
                        x=fn[month]['count'],
                        y=fn[month]['keywords'],
                        text=fn[month]['keywords'],
                        textposition='auto',  # position text outside bar
                        insidetextanchor='start',
                        marker={'color': fn[month]['colour']}
                    )
                ],
                name=month,  # This will tie the frame to a specific month
                layout=go.Layout(
                    xaxis=dict(range=[0, fn[months[0]]['count'].max()*1.42+100],
                               autorange=False,
                               visible=True,),
                    yaxis=dict(range=[-1, K[1]-K[0]+1],
                               visible=True,
                               autorange=False,
                               tickfont=dict(size=14)),
                    title=f'Amounts appear per month: {month}',

                ),
            )
            for month in months
        ]
    )
    # Update the figure layout with the initial title
    fig1.update_layout(
        sliders=[sliders_dict])
    return fig1


if __name__ == '__main__':
    app.run(debug=True)
