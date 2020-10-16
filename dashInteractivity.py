#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash('auth', external_stylesheets=external_stylesheets)  ### with css called 'external_stylesheets'

app.layout = html.Div([
    dcc.Input(id='my-id', value='Application Dash', type='text'),
    html.Div(id='my-div')
])


@app.callback(
    Output(component_id='my-div', component_property='children'),
    [Input(component_id='my-id', component_property='value')]
)
def update_output_div(input_value):
    return 'Vous venez de saisir "{}"'.format(input_value)


if __name__ == '__main__':
    app.run_server(debug=True)