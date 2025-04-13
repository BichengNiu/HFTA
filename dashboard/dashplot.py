import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import base64
import io

# Initialize the Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Store(id='data-store'),
        
    # Flex container for sidebar and graph
    html.Div([
        # Sidebar with upload and status
        html.Div([
            dcc.Upload(
                id='upload-data',
                children=html.Button('Upload File')
            ),
            html.Div(id='upload-status'),
        ], style={'width': '200px', 'padding': '10px'}),
        # Graph
        html.Div([
            dcc.Graph(id='time-series-plot', style={'width': '100%', 'height': '100%'})
        ], style={'flex': '1'})
    ], style={'display': 'flex', 'flex': '1'}),
    
    # Dropdown menu
    html.Div([
        dcc.Dropdown(id='variable-select', disabled=True),
    ], style={'padding': '10px'}),
    
], style={'display': 'flex', 'flexDirection': 'column', 'height': '100vh'})

# Function to parse the uploaded file and replace zeros with NaN
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename.lower():
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename.lower():
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            return None, 'Unsupported file type. Please upload a CSV or Excel file.'
    except Exception as e:
        return None, f'Error parsing file: {str(e)}'
    
    # Assume the first column or 'date' is the date column
    date_col = next((col for col in df.columns if col.lower() == 'date'), df.columns[0])
    df['date'] = pd.to_datetime(df[date_col])
    df.set_index('date', inplace=True)
    df['year'] = df.index.year
    df['week'] = df.index.isocalendar().week
    
    # Replace zeros with NaN in variable columns
    variables = [col for col in df.columns if col not in ['year', 'week']]
    df[variables] = df[variables].replace(0, np.nan)
    
    return df, 'File uploaded successfully'

# Callback to handle file upload and update dropdown
@app.callback(
    [
        Output('data-store', 'data'),
        Output('variable-select', 'options'),
        Output('variable-select', 'value'),
        Output('variable-select', 'disabled'),
        Output('upload-status', 'children')
    ],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)
def update_data(contents, filename):
    if contents is None:
        return None, [], None, True, 'Please upload a file'
    
    df, message = parse_contents(contents, filename)
    if df is None:
        return None, [], None, True, message
    
    data = df.to_json(date_format='iso', orient='split')
    variables = [col for col in df.columns if col not in ['year', 'week']]
    options = [{'label': var, 'value': var} for var in variables]
    return data, options, variables[0], False, message

# Callback to update the plot based on selected variable
@app.callback(
    Output('time-series-plot', 'figure'),
    [
        Input('variable-select', 'value'),
        Input('data-store', 'data')
    ]
)
def update_plot(selected_var, data):
    if data is None or selected_var is None:
        return go.Figure()
    
    df = pd.read_json(data, orient='split')
    current_year = df['year'].max()
    past_years = [y for y in df['year'].unique() if y < current_year][-4:]
    last_year = max(past_years)
    
    df_past = df[df['year'].isin(past_years)]
    df_current = df[df['year'] == current_year]
    df_last_year = df[df['year'] == last_year]
    stats = df_past.groupby('week')[selected_var].agg(['min', 'max', 'mean'])
    
    fig = go.Figure()
    
    # Gray band for min-max range
    fig.add_trace(go.Scatter(
        x=stats.index,
        y=stats['min'],
        mode='lines',
        line=dict(color='gray', width=0),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=stats.index,
        y=stats['max'],
        mode='lines',
        line=dict(color='gray', width=0),
        fill='tonexty',
        fillcolor='rgba(128,128,128,0.5)',
        showlegend=False
    ))

    # average line
    fig.add_trace(go.Scatter(
        x=stats.index,
        y=stats['mean'],
        mode='lines',
        line=dict(color='gray', dash='dot'),
        name='4-Year Average'
    ))
    
    # current year line
    fig.add_trace(go.Scatter(
        x=df_current['week'],
        y=df_current[selected_var],
        mode='lines',
        line=dict(color='red'),
        name=f'{current_year}'
    ))

    # last year line
    fig.add_trace(go.Scatter(
        x=df_last_year['week'],
        y=df_last_year[selected_var],
        mode='lines',
        line=dict(color='darkblue', width=2),
        name=f'{last_year}'
    ))

    fig.update_layout(
        title=f'Weekly Time Series for {selected_var}',
        xaxis_title='Week',
        yaxis_title=selected_var,
        legend=dict(x=1.05, y=1.0, xanchor='left', yanchor='top')
    )
    return fig

# Run the app
if __name__ == '__main__':
    app.run(debug=True)