
import os, sys, dash
import numpy as np
import pandas as pd


import seaborn as sns

import plotly.graph_objects as go
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import plotly.express as px

# =============================================================================
# 
# =============================================================================
import pandas as pd
data = pd.read_csv(r'C:\Users\Alvvv\Desktop\Digital lab\Projet dataviz\carData.csv')
data['Present_Price'].astype(int)
data['Selling_Price'].astype(int)
data['Kms_Driven'].astype(int)
data['Year'].astype(int)
data['Owner'].astype(int)
data.insert(2, "Age", 2020-data['Year'], True)
def score_to_numeric(x):
    if x=='CNG':
        return 3
    if x=='Diesel':
        return 2
    if x=='Petrol':
        return 1
data['Fuel_Type_num'] = data['Fuel_Type'].apply(score_to_numeric)
def score_to_numeric1(x):
    if x=='Automatic':
        return 0
    if x=='Manual':
        return 1
    
data['Transmission_num'] = data['Transmission'].apply(score_to_numeric1)
X= data[['Age','Kms_Driven','Fuel_Type_num','Transmission_num']]
Y=data['Selling_Price']

# =============================================================================
# 
# =============================================================================
secteur_dropdown = [
    {'label': "Regression linéaire", 'value': 'ols'},
    {'label': "Support Vector Machine", 'value': 'svm'},
]



app = dash.Dash()


app.config['suppress_callback_exceptions']=True
app.title = "App Title"


def build_tabs():
    return html.Div(
                id="tabs",
                className="tabs",
                children=[
                    dcc.Tabs(
                        id="app-tabs",
                        value="stats",
                        className="custom-tabs",
                        children=[
                            dcc.Tab(
                                id="tab1",
                                label="Statistiques",
                                value="stats",
                                className="custom-tab",
                                selected_className="custom-tab--selected",
                            ),                            
                            dcc.Tab(
                                id="tab2",
                                label="Regression",
                                value="regression",
                                className="custom-tab",
                                selected_className="custom-tab--selected",
                            ),
                        ],
                    )
                ],
            )

app.layout = html.Div(
    id="big-app-container",
    children=[
        html.Div(
            id="app-container",
            children=[
                build_tabs(),
                # Main app
                html.Div(id="app-content"),
            ],
        ),
    ],
)
 

@app.callback(
    Output('app-content', 'children'),
    [Input('app-tabs', 'value')],
)  

def app_body(tab):
    transparent = 'rgba(1, 1, 1, 0.0)'
    if tab=='stats':  
        
        temp = data['Transmission'].value_counts()        
        labels,values = temp.index.tolist(),temp.tolist()
        stats = go.Figure()
        stats.add_trace(go.Pie(labels=labels,values=values,pull=[0.05 for i in range(len(labels))]))
        stats.layout.update(
            title=dict(text="Répartition"),
            )
        
        temp = data['Fuel_Type'].value_counts()        
        labels,values = temp.index.tolist(),temp.tolist()
        stats1 = go.Figure()
        stats1.add_trace(go.Pie(labels=labels,values=values,pull=[0.05 for i in range(len(labels))]))
        stats1.layout.update(
            title=dict(text="Répartition"),
            )
        
        
        
        return html.Div([
                html.Div(
                    className='row',
                    children=[ 
                        html.Div(
                            className='twelve columns',
                            children=[
                                html.Div(
                                    className='six columns',
                                    children=[
                                        html.Div(
                                        className='row',
                                        children=[
                                            dcc.Graph(id='stats',
                                                      figure=stats)
                                            ]
                                        ), 
                                    ]
                                ),
                                html.Div(
                                    className='six columns',
                                    children=[
                                        html.Div(
                                        className='row',
                                        children=[
                                            dcc.Graph(id='stats1',
                                                      figure=stats1)
                                            ]
                                        ), 
                                    ]
                                )                                
                            ]
                        ),
                    ]
                )
            ]
        )                                                                            
    elif tab=="regression":
        return html.Div(
            className='row',
            children=[
                html.Div(
                    className='five columns',
                    children=[
                        html.Label(
                            ["Modèles", dcc.Dropdown(
                                id='modele',
                                options=secteur_dropdown,
                                value="ols",
                                style={'width': '100%'},
                                )
                            ],
                        )
                    ],
                ),
                html.Div(
                className='seven columns',
                children=[
                    html.Div(
                        id = 'content'    
                    )
                ]
            )
         ]   
    )
    
    

@app.callback(
    Output('content', 'children'),
    [Input('modele', 'value')],
) 
def function_ols(modele):
    if modele == 'ols':
        model = LinearRegression()
        result = model.fit(X, Y)
        r2 = model.score(X, Y)
        fig = px.scatter(data, x="Age", y="Present_Price", trendline="ols")
        
        return html.Div([
            html.Div(
                className="row",
                children=[
                    dcc.Graph(id = "fig", figure=fig)
                    ]
                ),
                html.Div(
                    className="row",
                    children=[
                        html.P(
                            "Le R2 du Regression lineaire est de {}%".format(round(r2*100, 2))
                        )
                    ]
                )
            ]            
        )              
        
    elif modele=='svm':
        model = SVR()
        result = model.fit(X, Y)
        r2 = result.score(X, Y)
        Xvr=result.predict(X)
        fig = px.scatter(data, x=Xvr, y="Present_Price", trendline="ols")
        return html.Div([
            html.Div(
                className="row",
                children=[
                    dcc.Graph(id = "fig", figure=fig)
                    ]
                ),
            
              html.Div(
                className="row",
                children=[
                    html.P(
                        "Le R2 du SVR est de {}%".format(round(r2*100, 2))
                        )
                    ]
                )
            ]
        )
        
        
        

if __name__ == '__main__':
    app.run_server()       