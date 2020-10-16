import dash
import dash_auth
import dash_core_components as dcc
import dash_html_components as html


VALID_USERNAME_PASSWORD_PAIRS = [
    ['anne', 'laure'],
    ['one','day']
]

#app = dash.Dash('auth')  ## without css
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash('auth', external_stylesheets=external_stylesheets)  ### with css called 'external_stylesheets'

auth = dash_auth.BasicAuth(
    app,
    VALID_USERNAME_PASSWORD_PAIRS
)


markdown_text = '''
### Dash and Markdown
Hello everyone !
You can write what you want here !!
'''

app.layout = html.Div([
    dcc.Markdown(children=markdown_text),
    
    html.Label('Dropdown'),
    dcc.Dropdown(
        options=[
            {'label': 'Taj Mahal Inde', 'value': 'TMI'},
            {'label': 'Big Ben Angleterre', 'value': 'BBA'},
            {'label': 'Grande Pyramide de Gizeh Egypte', 'value': 'GPGE'},
            {'label': 'Tour Eiffel France', 'value': 'TEF'},
            {'label': u'Le Colisée Italie', 'value': 'CI'}
        ],
        value='TMI'
    ),
            
    html.Label('Multi-Select Dropdown'),
    dcc.Dropdown(
        options=[
            {'label': 'Taj Mahal Inde', 'value': 'TMI'},
            {'label': 'Big Ben Angleterre', 'value': 'BBA'},
            {'label': 'Grande Pyramide de Gizeh Egypte', 'value': 'GPGE'},
            {'label': 'Tour Eiffel France', 'value': 'TEF'},
            {'label': u'Le Colisée Italie', 'value': 'CI'}
        ],
        value=['TMI', 'BBA'],
        multi=True
    ),
        
    html.Label('Radio Items'),
    dcc.RadioItems(
        options=[
            {'label': 'Taj Mahal Inde', 'value': 'TMI'},
            {'label': 'Big Ben Angleterre', 'value': 'BBA'},
            {'label': 'Grande Pyramide de Gizeh Egypte', 'value': 'GPGE'},
            {'label': 'Tour Eiffel France', 'value': 'TEF'},
            {'label': u'Le Colisée Italie', 'value': 'CI'}
        ],
        value='GPGE'
    ),
            
    html.Label('Checkboxes'),
    dcc.Checklist(
        options=[
            {'label': 'Taj Mahal Inde', 'value': 'TMI'},
            {'label': 'Big Ben Angleterre', 'value': 'BBA'},
            {'label': 'Grande Pyramide de Gizeh Egypte', 'value': 'GPGE'},
            {'label': 'Tour Eiffel France', 'value': 'TEF'},
            {'label': u'Le Colisée Italie', 'value': 'CI'}
        ],
        value=['TMI', 'TEF']
    ),
            
    html.Label('Text Box'),
    dcc.Input(value='Hello Word', type='text')
]

)


if __name__ == '__main__':
    app.run_server(debug=True)
