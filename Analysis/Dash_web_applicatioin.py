# -*- coding: utf-8 -*-
import pandas as pd
import os
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_daq as daq
import plotly.plotly as py
import plotly.graph_objs as go
from datetime import datetime as dt
import pickle
import numpy as np


def RADIANS(x):
    rad = x * np.pi / 180
    return rad
def RADIANS_TO_KM(y):
    distance_to_km = 111.045 * 180 * y / np.pi
    return distance_to_km
def HAVERSINE(lat1, long1, lat2, long2):
    distance = RADIANS_TO_KM(np.arccos(np.cos(RADIANS(lat1)) * np.cos(RADIANS(lat2)) * np.cos(RADIANS(long1) - RADIANS(long2)) + np.sin(RADIANS(lat1)) * np.sin(RADIANS(lat2))))
    return distance
###########################################################################
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
mapbox_access_token='pk.eyJ1Ijoic2hhbmVub2x1bmExMTA2IiwiYSI6ImNqdmdocHg3dDAzdG8zeW1ncGs4ODY1ZWoifQ.ImCgzxvA8o8saZp2KjFODg'
###########################################################################


app.layout = html.Div(
    style={
    'background-image':'url("/assets/taxi_black.png")'
    },
    
    children=[
    html.H1(children='A taxi duration ETA app',
        style={
                'textAlign': 'center',
                'color': '#1ab6ff',
                'font-weight':'bold'
            }),

    html.Div([
        html.Div([
        html.H4('Travel Date',style={'color':'#1ab6ff'}),
        dcc.DatePickerSingle(
            id='date',
            date=dt(2016, 5, 10)
        ),
        html.H4('Temperature',style={'color':'#1ab6ff'}),
        dcc.Input(id='temp', value=60, type='number')],
        style={'width': '20%', 'display': 'inline-block'}),

        html.Div([
        html.H4('Pickup Latitude',style={'color':'#1ab6ff'}),
        dcc.Input(id='pickup_latitude', value=40.75, type='number'),
        html.H4('Pickup Longitude',style={'color':'#1ab6ff'}),
        dcc.Input(id='pickup_longitude', value=-73.99, type='number')],
        style={'width': '20%', 'display': 'inline-block'}),

        html.Div([
        html.H4('Dropoff Latitude',style={'color':'#1ab6ff'}),
        dcc.Input(id='dropoff_latitude', value=40.75, type='number'),
        html.H4('Dropoff Longitude',style={'color':'#1ab6ff'}),
        dcc.Input(id='dropoff_longitude', value=-73.95, type='number')],
        style={'width': '20%', 'float': 'left', 'display': 'inline-block'})],

        style={'width': '96%','padding-left':'23%', 'padding-right':'1%'}

    ),

    html.Div([
        dcc.Graph(id='map')],
        style={'width': '96%','padding-left':'20%', 'padding-right':'1%'}
    ),  

    html.Div([
    html.H4(children='Estimated Travel Duration: ',
        style={
                'color': '#1ab6ff',
                'font-weight':'bold'}), ##FF3D55
    # dcc.Textarea(
    #     id='estimate_time',
    #     placeholder='Enter a value...',
    #     value='This is a TextArea component',
    #     style={'width': '20%'})
    daq.LEDDisplay(
        id='estimate_time',
        color='#1ab6ff',
        value="100.00")   
    ],style={'width': '96%','padding-left':'40%', 'padding-right':'1%'})

])


@app.callback(
    [Output(component_id='map', component_property='figure'),
    Output(component_id='estimate_time', component_property='value')],
    [Input(component_id='pickup_latitude', component_property='value'),
    Input(component_id='pickup_longitude', component_property='value'),
    Input(component_id='dropoff_latitude', component_property='value'),
    Input(component_id='dropoff_longitude', component_property='value'),
    Input(component_id='date', component_property='date'),
    Input(component_id='temp', component_property='value')])

def update_graph(p_lat,p_lon,d_lat,d_lon,date,temp):
    # create test data
    df_test=pd.read_csv('../Data/test_weather.csv',nrows = 1)
    df_test[['pickup_latitude']]=p_lat
    df_test[['pickup_longitude']]=p_lon
    df_test[['dropoff_latitude']]=d_lat
    df_test[['dropoff_longitude']]=d_lon
    df_test[['temp']]=temp
    if p_lat==d_lat and p_lon==d_lon:
        df_test['distancce_in_km'] = 0
    else:
        df_test['distancce_in_km'] = HAVERSINE(df_test.pickup_latitude, df_test.pickup_longitude, df_test.dropoff_latitude, df_test.dropoff_longitude)
    # create `day_of_week` and `pickup_hour` column
    df_test['pickup_datetime'] = date
    df_test['pickup_datetime'] = pd.to_datetime(df_test['pickup_datetime'])
    df_test['day_of_week'] = df_test.pickup_datetime.dt.weekday
    # Add pick-up hour
    df_test['pickup_hour'] = df_test.pickup_datetime.dt.hour

    # get X
    X=df_test[['passenger_count', 'pickup_longitude', 'pickup_latitude','dropoff_longitude', 'dropoff_latitude',
            'temp', 'distancce_in_km','day_of_week', 'pickup_hour']] #'vendor_id',
    # get saved model
    with open('saved/xgb.pickle', 'rb') as f:
        xgb = pickle.load(f)
    time = xgb.predict(X)

    return {
        'data': [go.Scattermapbox(
            lat=[str(p_lat),str(d_lat)],
            lon=[str(p_lon),str(d_lon)],
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=10,
                opacity=1,
                # 'line': {'width': 0.5, 'color': 'white'}
            ),
            text=['Pickup Location','Dropoff Location'],
        )],
        'layout': go.Layout(
            # title='NYC Taxi Map',
            width=800,
            height=800,
            autosize=True,
            hovermode='closest',
            showlegend=False,
            paper_bgcolor = 'rgba(0,0,0,0)',
            mapbox=go.layout.Mapbox(
                accesstoken=mapbox_access_token,
                bearing=0,
                center=go.layout.mapbox.Center(
                    lat=40.75,
                    lon=-73.99
                ),
                pitch=0,
                zoom=11,
                style='light'
            ),
         )
         }, u'{:.2f}'.format(time[0]) # This travel will likely cost you 



if __name__ == '__main__':
    app.run_server(debug=True)