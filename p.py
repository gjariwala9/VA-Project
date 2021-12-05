import base64
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import joblib
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Output, Input, State
from flask import Response
from matplotlib.widgets import Button, Slider
from pathlib import Path
from plotly.subplots import make_subplots
from custom_function import *
from typing import Tuple

model = joblib.load('finalModel.joblib')
scaler = pickle.load(open('scaler.pkl', 'rb'))
le = pickle.load(open('labelEncoder.pkl', 'rb'))

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Rainfall Prediction in Australia"
app.config.suppress_callback_exceptions = True

df = pd.read_csv('dataset/weatherAUS-processed.csv')
aus_cities = pd.read_csv('dataset/au_cities.csv')
df = pd.merge(df, aus_cities, on='Location', how='left')


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dbc.Row([
        html.A(html.H1(children='Australian Rainfall - Visualization Dashboard', style={
            "font-style": "italic",
            "font-family": "Audrey"}), href="/", style={'text-dacoration': 'none', 'color': 'yellow'}),
    ],
        justify="center",
        style={"margin-top": "0", "margin-bottom": "1px", "color": "#ffffff", "background-color": "#000000"}
    ),
    html.Div(id='page-content', style={
        "background-image": "url('https://www.basicplanet.com/wp-content/uploads/2017/01/Countries-with-Most-Rainfall-in-the-World.jpg')",
        "padding-top": "50px"})
])

index_page = dbc.Container([
    html.Title("Australian Rainfall Prediction"),
    dbc.Row([
        dbc.Col(
            dbc.Card(
                [
                    dbc.CardImg(src=app.get_asset_url('history.png'), top=True),
                    dbc.CardBody(
                        [
                            html.H4("Rainfall History", className="card-title"),
                            html.P(
                                "Learn about Rainfall in Australia. Juxtapose the demographics.",
                                className="card-text",
                            ),
                            dcc.Link(dbc.Button("Click to Visualize", color="warning"), href="/rainfall-history",
                                     refresh=True),

                        ]
                    ),
                ],
                style={"width": "18rem", "height": "23rem", "background-color": "#000000", "color": "#ffffff"},
            ),
        ),
        dbc.Col(
            dbc.Card(
                [
                    dbc.CardImg(src=app.get_asset_url('location.jpg'), top=True, style={"height": "45%"}),
                    dbc.CardBody(
                        [
                            html.H4("Location", className="card-title"),
                            html.P(
                                "Visiting a place and wanna know about the rainfall there? We can help!",
                                className="card-text",
                            ),
                            dcc.Link(dbc.Button("Click to Visualize", color="warning"), href="/location",
                                     refresh=True),
                        ]
                    ),
                ],
                style={"width": "18rem", "height": "23rem", "background-color": "#000000", "color": "#ffffff"},
            ),
        )
    ], style={"position": "relative", "left": "110px"}),

    dbc.Row([
        dbc.Col(
            dbc.Card(
                [
                    dbc.CardImg(src=app.get_asset_url('visualization.jpg'), top=True),
                    dbc.CardBody(
                        [
                            html.H4("Rainfall Visualization", className="card-title"),
                            html.P(
                                "Did you know that we consider all the parameters to learn about past rainfall? Use these parameters as per your will to cluster similar rainfall!",
                                className="card-text",
                            ),
                            dcc.Link(dbc.Button("Click to Visualize", color="warning"), href="/rainfall-visualization",
                                     refresh=True),
                        ]
                    ),
                ],
                style={"width": "18rem", "background-color": "#000000", "color": "#ffffff"},
            ),
        ),
        dbc.Col(
            dbc.Card(
                [
                    dbc.CardImg(src=app.get_asset_url('network.jpg'), top=True),
                    dbc.CardBody(
                        [
                            html.H4("Predict Rainfall", className="card-title"),
                            html.P(
                                "We provide you a ready made ML model and allow you to tweak thedemographics to predict future rainfall! The rainfall are visualized using a Gauge plot",
                                className="card-text",
                            ),
                            dcc.Link(dbc.Button("Click to Predict", color="warning"), href="/rainfall-predict",
                                     refresh=True),
                        ]
                    ),
                ],
                style={"width": "18rem", "background-color": "#000000", "color": "#ffffff"},
            ),
        ),
    ],
        # justify="center,
        style={"position": "relative", "left": "110px", "margin-top": "20px"}
    ),
],
    style={"margin-top": 0, "padding-top": "50px"}
)


def fetch_years():
    lst = df['year'].unique()
    lst = lst.tolist()
    lst.sort()
    years = []
    for year in lst:
        years.append({"label": year, "value": year})
    return years


def fetch_numeric_columns():
    num_cols = []
    numeric_cols = get_numeric_columns(df)
    for col in numeric_cols:
        if col == 'year':
            continue
        num_cols.append({"label": col, "value": col})
    return num_cols


def fetch_directions():
    lst = df['WindDir9am'].unique()
    lst = lst.tolist()
    lst.sort()
    directions = []
    for direction in lst:
        directions.append({"label": direction, "value": direction})
    return directions


def fetch_cities():
    lst = df['Location'].unique()
    lst = lst.tolist()
    lst.sort()
    cities = []
    for city in lst:
        cities.append({"label": city, "value": city})
    return cities


def get_layout_for_tab1():
    layout = html.Div([
        dbc.Row([
            html.H3(children='Demogrphic Visualization'),
        ],
            style={"margin-top": "50px", "margin-bottom": "20px", "color": "#ffffff", "background-color": "#000000",
                   "padding-left": "1%"}
        ),
        html.Div([
            dbc.FormGroup([
                dbc.Label("Select Year"),
                dcc.Dropdown(id="dropdown_years_chart", value=1, options=fetch_years()),

                dbc.Label("Select Chart"),
                dcc.Dropdown(id="dropdown_chart", value=2, options=[{"label": "Bar Chart", "value": "bar"},
                                                                    {"label": "line Chart", "value": "line"},
                                                                    {"label": "Bar-Line", "value": "bar-line"}]),
                dbc.Label("Select feature"),
                dcc.Dropdown(id="dropdown_feature_chart", value=3, options=fetch_numeric_columns()),
                html.Br()
            ]),
            dbc.Button('Show Chart', id='button_chart', color='warning', style={'margin-bottom': '1em'},
                       block=True),
            dbc.Row([
                dbc.Col(dcc.Graph(id='viz2')),
            ]),
        ])
    ])
    return layout


def get_layout_for_tab2():
    layout = html.Div([
        dbc.Row([
            html.H3(children='Trend'),
        ],
            style={"margin-top": "50px", "margin-bottom": "20px", "color": "#ffffff", "background-color": "#000000",
                   "padding-left": "1%"}
        ),
        html.Div([
            dbc.FormGroup([
                dbc.Label("Select city"),
                dcc.Dropdown(id="dropdown_cities", value=1, options=fetch_cities()),
                dbc.Label("Select feature"),
                dcc.Dropdown(id="dropdown_feature_trend", value=2, options=fetch_numeric_columns()),
                html.Br()
            ]),
            dbc.Button('Show Chart', id='button_trend', color='warning', style={'margin-bottom': '1em'},
                       block=True),
            dbc.Row([
                dbc.Col(dcc.Graph(id='viz3')),
            ]),
        ])
    ])

    return layout


def get_layout_for_tab3():
    layout = html.Div([
        dbc.Row([
            html.H3(children='Correlation'),
        ],
            style={"margin-top": "50px", "margin-bottom": "20px", "color": "#ffffff", "background-color": "#000000",
                   "padding-left": "1%"}
        ),
        html.Div([
            dbc.FormGroup([

                dbc.Label("Select Chart"),
                dcc.RadioItems(id="dropdown_chart_corr", value='pair', options=[
                    {"label": "Pair Plot", "value": "pair"},
                    {"label": "Heat-Map", "value": "heat-map"}, ],
                               labelStyle={'display': 'inline-block', 'padding-right': '15px'}),
                dbc.Label("Select feature"),
                dcc.Dropdown(id="dropdown_feature_corr", value=2, options=fetch_numeric_columns(), multi=True),
                html.Br()
            ]),
            dbc.Button('Show Chart', id='button_corr', color='warning', style={'margin-bottom': '1em'},
                       block=True),
            dbc.Row([
                dbc.Col(dcc.Graph(id='viz4')),
            ]),
        ])
    ])

    return layout


page1 = dbc.Container([
    dcc.Tabs([
        dcc.Tab(label='Demographic Visualization', children=[
            get_layout_for_tab1()
        ]),
        dcc.Tab(label='Trend Analysis', children=[
            get_layout_for_tab2()
        ]),
        dcc.Tab(label='Feature Correlation', children=[
            get_layout_for_tab3()
        ])
    ], style={'font-style': 'italic', 'color': 'red', 'background-color': 'black'})
], style={"background-color": "#ffffff"})

page3 = dbc.Container([
    dbc.Row([
        html.H3(children='Rainfall Visualization'),
    ],
        style={"margin-top": "50px", "margin-bottom": "20px", "color": "#ffffff", "background-color": "#000000",
               "padding-left": "1%"}
    ),
    html.Div([
        dbc.FormGroup([
            dbc.Label("Select Cities"),
            dcc.Dropdown(id="dropdown_cities_3", value=1, options=fetch_cities(), multi=True),

            dbc.Label("Select feature"),
            dcc.Dropdown(id="dropdown_feature_3", value=2, options=fetch_numeric_columns()),

            html.Br()
        ]),
        dbc.Button('Show Visualization', id='button_3', color='warning', style={'margin-bottom': '1em'},
                   block=True),
        dbc.Row([
            dbc.Col(dcc.Graph(id='viz_slider_autoplay')),
        ]),
        dcc.Slider(
            id="auto_slider",
            min=2007,
            max=2017,
            step=None,
            marks={
                2007: '2007',
                2008: '2008',
                2009: '2009',
                2010: '2010',
                2011: '2011',
                2012: '2012',
                2013: '2013',
                2014: '2014',
                2015: '2015',
                2016: '2016',
                2017: '2017'
            },
            value=2012
        )
    ]),
], style={"background-color": "#ffffff"})

page4 = dbc.Container([
    dbc.Row([
        html.H3(children='Rainfall Prediction'),
    ],
        style={"margin-top": "50px", "margin-bottom": "20px", "color": "#ffffff", "background-color": "#000000",
               "padding-left": "1%"}
    ),
    html.Div([
        dbc.FormGroup([
            dbc.Label("Select City"),
            dcc.Dropdown(id="dropdown_city_4", value='Cobar', options=fetch_cities()),

            dbc.Label("Enter Minimum Temperature"),
            dbc.Input(id="minTemp", type="number", min=-50, max=50, step=0.1, value=17.9),

            dbc.Label("Enter Maximum Temperature"),
            dbc.Input(id="maxTemp", type="number", min=-50, max=50, step=0.1, value=35.2),

            dbc.Label("Enter Rainfall in cm"),
            dbc.Input(id="rain", type="number", min=0, max=100, step=0.1, value=0.0),

            dbc.Label("Enter Evaporation"),
            dbc.Input(id="evaporation", type="number", min=0, max=100, step=0.1, value=12.0),

            dbc.Label("Enter Sunshine"),
            dbc.Input(id="sunshine", type="number", min=0, max=100, step=0.1, value=12.3),

            dbc.Label("Select Wind Gust Direction"),
            dcc.Dropdown(id="windGustDir", value='SSW', options=fetch_directions()),

            dbc.Label("Enter WindGustSpeed"),
            dbc.Input(id="windGustSpeed", type="number", min=0, max=100, step=0.1, value=48.0),

            dbc.Label("Select Wind Direction at 9AM"),
            dcc.Dropdown(id="windDir9am", value='ENE', options=fetch_directions()),

            dbc.Label("Select Wind Direction at 3PM"),
            dcc.Dropdown(id="windDir3pm", value='SW', options=fetch_directions()),

            dbc.Label("Enter Wind Speed at 9AM"),
            dbc.Input(id="windSpeed9am", type="number", min=0, max=100, step=0.1, value=6.0),

            dbc.Label("Enter Wind Speed at 3PM"),
            dbc.Input(id="windSpeed3pm", type="number", min=0, max=100, step=0.1, value=20.0),

            dbc.Label("Enter Humidity at 9AM"),
            dbc.Input(id="humidity9am", type="number", min=0, max=100, step=0.1, value=20.0),

            dbc.Label("Enter Humidity at 3PM"),
            dbc.Input(id="humidity3pm", type="number", min=0, max=100, step=0.1, value=13.0),

            dbc.Label("Enter Pressure at 9AM"),
            dbc.Input(id="pressure9am", type="number", min=0, max=10000, step=0.1, value=1006.3),

            dbc.Label("Enter Pressure at 3PM"),
            dbc.Input(id="pressure3pm", type="number", min=0, max=10000, step=0.1, value=1004.4),

            dbc.Label("Enter Cloud at 9AM"),
            dbc.Input(id="cloud9am", type="number", min=0, max=100, step=0.1, value=2.0),

            dbc.Label("Enter CLoud at 3PM"),
            dbc.Input(id="cloud3pm", type="number", min=0, max=100, step=0.1, value=5.0),

            dbc.Label("Enter Temperature at 9AM"),
            dbc.Input(id="temp9am", type="number", min=0, max=100, step=0.1, value=26.6),

            dbc.Label("Enter Temperature at 3PM"),
            dbc.Input(id="temp3pm", type="number", min=0, max=100, step=0.1, value=33.4),

            dbc.Label("Did it rain today?"),
            dcc.Dropdown(id="rainToday", value='No', options=[{'label': 'Yes', 'value': 'Yes'},
                                                              {'label': 'No', 'value': 'No'}]),

            html.Br()
        ]),
        dbc.Button('Predict', id='predictBtn', color='warning', style={'margin-bottom': '1em'},
                   block=True),

        html.Div(id='prediction'),
    ]),
], style={"background-color": "#ffffff"})

error_page = dbc.Container([
    html.Div([
        dbc.Row([
            html.H3(children='Coming Soon!'),
        ],
            # justify="center",
            style={"margin-bottom": "20px", "color": "#ffffff", "background-color": "#000000", "padding-left": "1%"}
        )
    ]),
    html.Div([
        dbc.Row([
            html.H6(children='The page is under development phase. Please come back after some time.',
                    style={"color": "red"}),
        ])
    ])
], style={"height": "100vh"})

page2 = dbc.Container([
    html.Div([
        dbc.Row([
            html.H3(children='Rainfall: At your Destination'),
        ],
            # justify="center",
            style={"margin-bottom": "20px", "color": "#ffffff", "background-color": "#000000", "padding-left": "1%"}
        ),
        dbc.Tabs([
            dbc.Tab(label='Custom Visualization', children=[
                dbc.FormGroup([
                    dbc.Label("Select Cities"),
                    dcc.Dropdown(id="dropdown_city_location", value=1, options=fetch_cities(), multi=True),
                    dbc.Label("Select feature"),
                    dcc.Dropdown(id="dropdown_feature_location", value=2, options=fetch_numeric_columns()),
                    html.Br()
                ]),
                dcc.RangeSlider(
                    id='range_slider_year',
                    min=2007,
                    max=2017,
                    step=None,
                    marks={
                        2007: '2007',
                        2008: '2008',
                        2009: '2009',
                        2010: '2010',
                        2011: '2011',
                        2012: '2012',
                        2013: '2013',
                        2014: '2014',
                        2015: '2015',
                        2016: '2016',
                        2017: '2017'
                    },
                    value=[2007, 2017]
                ),
                dbc.Button('Show Visualization', id='button_location', color='warning', style={'margin-bottom': '1em'},
                           block=True),
                dcc.Tabs([
                    dcc.Tab(label='Rainfall Trends', children=[
                        dbc.Row([
                            dbc.Col(dcc.Graph(id='viz_location')),
                        ])
                    ]),
                    dcc.Tab(label='Rainfall Distribution', children=[
                        dbc.Row([
                            dbc.Col(dcc.Graph(id='viz_directions')),
                        ])
                    ])
                ], style={'font-style': 'italic', 'color': 'red', 'background-color': 'black'})
            ]),
            dbc.Tab(label='Holistic View', children=[
                dbc.FormGroup([
                    dbc.Label("Select feature"),
                    dcc.Dropdown(id="ddn_feature_ch", value=2, options=fetch_numeric_columns()),
                    html.Br()
                ]),
                dcc.RangeSlider(
                    id='year_slider_ch',
                    min=2007,
                    max=2017,
                    step=None,
                    marks={
                        2007: '2007',
                        2008: '2008',
                        2009: '2009',
                        2010: '2010',
                        2011: '2011',
                        2012: '2012',
                        2013: '2013',
                        2014: '2014',
                        2015: '2015',
                        2016: '2016',
                        2017: '2017'
                    },
                    value=[2007, 2017]
                ),
                dbc.Button('Show Visualization', id='button_ch', color='warning', style={'margin-bottom': '1em'},
                           block=True),
                html.Br(),
                dbc.Col(dcc.Graph(id='viz_choropleth')),
            ])
        ]),
        # dbc.Row([
        #     dbc.Col(dcc.Graph(id='viz_location')),
        # ]),
        # dbc.Row([
        #     dbc.Col(dcc.Graph(id='viz_directions')),
        # ]),
    ]),

], style={"background-color": "#ffffff"})
#
# # Setting Web Layout
# page = dbc.Container([
#     # dbc.Row([
#     #     html.H1(children='Australian Rainfall - Visualization Dashboard'),
#     # ],
#     #     justify="center",
#     #     style={"margin-top": "50px", "margin-bottom": "20px", "color": "#ffffff", "background-color": "#000000"}
#     # ),
#     html.Div([
#         dbc.FormGroup([
#             dbc.Label("Select Year"),
#             dcc.Dropdown(id="dropdown_years", value=1, options=fetch_years()),
#             dbc.Label("Select feature"),
#             dcc.Dropdown(id="dropdown_feature", value=2, options=fetch_numeric_columns()),
#             html.Br()
#         ]),
#         dbc.Button('Show Map', id='button_map', color='success', style={'margin-bottom': '1em'},
#                    block=True),
#         dbc.Row([
#             dbc.Col(dcc.Graph(id='viz1')),
#         ]),
#     ], style={"margin-top": "2px"}),
#     dbc.Row([
#         html.H3(children='Demogrphic Visualization'),
#     ],
#         # justify="center",
#         style={"margin-top": "50px", "margin-bottom": "20px", "color": "#ffffff", "background-color": "#000000",
#                "padding-left": "1%"}
#     ),
#     html.Div([
#         dbc.FormGroup([
#             dbc.Label("Select Year"),
#             dcc.Dropdown(id="dropdown_years_chart", value=1, options=fetch_years()),
#
#             dbc.Label("Select Chart"),
#             dcc.Dropdown(id="dropdown_chart", value=2, options=[{"label": "Bar Chart", "value": "bar"},
#                                                                 {"label": "line Chart", "value": "line"},
#                                                                 {"label": "Bar-Line", "value": "bar-line"}]),
#             dbc.Label("Select feature"),
#             dcc.Dropdown(id="dropdown_feature_chart", value=3, options=fetch_numeric_columns()),
#             html.Br()
#         ]),
#         dbc.Button('Show Chart', id='button_chart', color='warning', style={'margin-bottom': '1em'},
#                    block=True),
#         dbc.Row([
#             dbc.Col(dcc.Graph(id='viz2')),
#         ]),
#     ]),
#     dbc.Row([
#         html.H3(children='Trend'),
#     ],
#         # justify="center",
#         style={"margin-top": "50px", "margin-bottom": "20px", "color": "#ffffff", "background-color": "#000000",
#                "padding-left": "1%"}
#     ),
#     html.Div([
#         dbc.FormGroup([
#             dbc.Label("Select city"),
#             dcc.Dropdown(id="dropdown_cities", value=1, options=fetch_cities()),
#
#             # dbc.Label("Select Chart"),
#             # dcc.Dropdown(id="dropdown_chart", value=2, options=[{"label": "Bar Chart", "value": "bar"},
#             #                                                {"label": "line Chart", "value": "line"}]),
#             dbc.Label("Select feature"),
#             dcc.Dropdown(id="dropdown_feature_trend", value=2, options=fetch_numeric_columns()),
#             html.Br()
#         ]),
#         dbc.Button('Show Chart', id='button_trend', color='warning', style={'margin-bottom': '1em'},
#                    block=True),
#         dbc.Row([
#             dbc.Col(dcc.Graph(id='viz3')),
#         ]),
#     ]),
#     dbc.Row([
#         html.H3(children='Correlation'),
#     ],
#         # justify="center",
#         style={"margin-top": "50px", "margin-bottom": "20px", "color": "#ffffff", "background-color": "#000000",
#                "padding-left": "1%"}
#     ),
#     html.Div([
#         dbc.FormGroup([
#             # dbc.Label("Select city"),
#             # dcc.Dropdown(id="dropdown_cities", value=1, options=fetch_cities()),
#
#             dbc.Label("Select Chart"),
#             dcc.Dropdown(id="dropdown_chart_corr", value=1, options=[{"label": "Heat-Map", "value": "heat-map"},
#                                                                      {"label": "Pair Plot", "value": "pair"}]),
#             dbc.Label("Select feature"),
#             dcc.Dropdown(id="dropdown_feature_corr", value=2, options=fetch_numeric_columns(), multi=True),
#             html.Br()
#         ]),
#         dbc.Button('Show Chart', id='button_corr', color='warning', style={'margin-bottom': '1em'},
#                    block=True),
#         dbc.Row([
#             dbc.Col(dcc.Graph(id='viz4')),
#         ]),
#     ]),
#
# ], style={"background-color": "#ffffff"})


def cat_to_var(var, lst):
    inputs = []
    for i in lst:
        if i == var:
            inputs.append(1.0)
        else:
            inputs.append(0.0)
    return inputs


@app.callback(
    dash.dependencies.Output('prediction', 'children'),
    [Input('predictBtn', 'n_clicks')],
    [State('dropdown_city_4', 'value'),
     State('minTemp', 'value'),
     State('maxTemp', 'value'),
     State('rain', 'value'),
     State('evaporation', 'value'),
     State('sunshine', 'value'),
     State('windGustDir', 'value'),
     State('windGustSpeed', 'value'),
     State('windDir9am', 'value'),
     State('windDir3pm', 'value'),
     State('windSpeed9am', 'value'),
     State('windSpeed3pm', 'value'),
     State('humidity9am', 'value'),
     State('humidity3pm', 'value'),
     State('pressure9am', 'value'),
     State('pressure3pm', 'value'),
     State('cloud9am', 'value'),
     State('cloud3pm', 'value'),
     State('temp9am', 'value'),
     State('temp3pm', 'value'),
     State('rainToday', 'value'),
     ]
)
def predictRainfall(n_clicks, city, minTemp, maxTemp, rainfall, evaporation, sunshine, windGustDir, windGustSpeed,
                    windDir9am, windDir3pm, windSpeed9am, windSpeed3pm, humidity9am, humidity3pm, pressure9am,
                    pressure3pm,
                    cloud9am, cloud3pm, temp9am, temp3pm, rainToday):
    if n_clicks:
        inputs = []
        inputs.append(minTemp)
        inputs.append(maxTemp)
        inputs.append(rainfall)
        inputs.append(evaporation)
        inputs.append(sunshine)
        inputs.append(windGustSpeed)
        inputs.append(windSpeed9am)
        inputs.append(windSpeed3pm)
        inputs.append(humidity9am)
        inputs.append(humidity3pm)
        inputs.append(pressure9am)
        inputs.append(pressure3pm)
        inputs.append(cloud9am)
        inputs.append(cloud3pm)
        inputs.append(temp9am)
        inputs.append(temp3pm)

        if rainToday == 'No':
            inputs.append(1)
            inputs.append(0)
        else:
            inputs.append(0)
            inputs.append(1)

        city_lst = ['AliceSprings', 'Brisbane', 'Cairns', 'Canberra', 'Cobar',
                    'CoffsHarbour', 'Darwin', 'Hobart', 'Melbourne',
                    'MelbourneAirport', 'Mildura', 'Moree', 'MountGambier',
                    'NorfolkIsland', 'Nuriootpa', 'Perth', 'PerthAirport', 'Portland',
                    'Sale', 'Sydney', 'SydneyAirport', 'Townsville', 'WaggaWagga',
                    'Watsonia', 'Williamtown', 'Woomera']

        inputs.extend(cat_to_var(city, city_lst))

        wind_direction_lst = ['E', 'ENE', 'ESE', 'N', 'NE', 'NNE', 'NNW', 'NW', 'S', 'SE', 'SSE',
                              'SSW', 'SW', 'W', 'WNW', 'WSW']

        inputs.extend(cat_to_var(windGustDir, wind_direction_lst))
        inputs.extend(cat_to_var(windDir9am, wind_direction_lst))
        inputs.extend(cat_to_var(windDir3pm, wind_direction_lst))

        data = [inputs]
        data = scaler.transform(data)
        result = model.predict(data)[0]
        if result == 0:
            return html.Div([
                html.H3('Its going to be sunny. Enjoy your day!!! :)'),
                html.Img(src=app.get_asset_url('sun.png'),
                         style={"width": "300px", "height": "300px", 'position': 'relative', 'left': '100px'})
            ], style={'position': 'relative', 'left': '300px'})
        else:
            return html.Div([
                html.H3('Its going to rain. Please carry your umbrella with you!!! :)'),
                html.Img(src=app.get_asset_url('rainfall.jpg'),
                         style={"width": "300px", "height": "300px", 'position': 'relative', 'left': '100px'})
            ], style={'position': 'relative', 'left': '300px'})

    return {}


@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/rainfall-history':
        return page1
    elif pathname == '/location':
        return page2
    elif pathname == '/rainfall-predict':
        return page4
    elif pathname == '/rainfall-visualization':
        return page3
    else:
        return index_page


@app.callback(
    dash.dependencies.Output('viz_choropleth', 'figure'),
    [Input('button_ch', 'n_clicks')],
    [State('year_slider_ch', 'value'),
     State('ddn_feature_ch', 'value'),
     ]
)
def update_vis1(n_clicks, year, feature):
    if n_clicks:
        df_map = df[df['year'].isin(year)]
        fig = None
        df_map = df_map.groupby('Location', as_index=False)[feature].mean()
        df_map = pd.merge(df_map, aus_cities, on='Location', how='left')
        fig = px.scatter_geo(df_map,
                             lat=df_map.lat,
                             lon=df_map.lng,
                             projection="natural earth",
                             locationmode="country names",
                             size=df_map[feature],
                             hover_data={'lat': False, 'lng': False},
                             color=df_map.Location
                             )
        fig.update_layout(
            title='Rainfall in Australia',
            geo=dict(
                projection_scale=0.15,  # this is kind of like zoom
                center=dict(lat=-25, lon=135),  # this will center on the point
                lataxis=dict(range=[-28, -22]),
                lonaxis=dict(range=[130, 140])
            )
        )
        return fig

    return {}


@app.callback(
    dash.dependencies.Output('viz_slider_autoplay', 'figure'),
    [Input('button_3', 'n_clicks'), Input('auto_slider', 'value')],
    [State('dropdown_cities_3', 'value'),
     State('dropdown_feature_3', 'value'),
     ]
)
def update_vis_3_1(n_clicks, year, cities, feature):
    if n_clicks:
        fig = None
        df_autoplay = df[df['Location'].isin(cities)]
        df_autoplay = df_autoplay[df_autoplay['year'] == year]
        df_autoplay = df_autoplay.groupby(['Location', 'RainTomorrow'], as_index=False)[feature].mean()

        fig = px.bar(df_autoplay, x='Location', y=feature, color='RainTomorrow', barmode="group")

        return fig

    return {}


@app.callback(
    dash.dependencies.Output('viz2', 'figure'),
    [Input('button_chart', 'n_clicks')],
    [State('dropdown_years_chart', 'value'),
     State('dropdown_chart', 'value'),
     State('dropdown_feature_chart', 'value'),
     ]
)
def update_vis2(n_clicks, year, chart, feature):
    if n_clicks:
        fig = None
        df_year = df.loc[df['year'] == year]
        df_year = df_year.groupby('Location', as_index=False)[feature].mean()
        if chart == 'bar':
            fig = px.bar(df_year, 'Location', feature)
        elif chart == 'line':
            fig = px.line(df_year, 'Location', feature)
        elif chart == 'bar-line':
            # Creating Bar Chart
            fig = px.bar(df_year, 'Location', feature)

            # Adding line chart on bar chart
            fig.add_trace(
                go.Scatter(x=df_year['Location'], y=df_year[feature])
            )
        return fig

    return {}


@app.callback(
    dash.dependencies.Output('viz3', 'figure'),
    [Input('button_trend', 'n_clicks')],
    [State('dropdown_cities', 'value'),
     State('dropdown_feature_trend', 'value'),
     ]
)
def update_vis3(n_clicks, city, feature):
    if n_clicks:
        fig = None
        df_city = df.loc[df['Location'] == city]
        df_city = df_city.groupby('year', as_index=False)[feature].mean()

        fig = px.line(df_city, 'year', feature)

        return fig

    return {}


@app.callback(
    dash.dependencies.Output('viz4', 'figure'),
    [Input('button_corr', 'n_clicks')],
    [
        State('dropdown_chart_corr', 'value'),
        State('dropdown_feature_corr', 'value'),
    ]
)
def update_vis4(n_clicks, chart, feature):
    if n_clicks:
        fig = None
        df_corr = df[feature]
        df_pair = df_corr.copy()
        df_pair['RainTomorrow'] = df['RainTomorrow']
        if chart == 'heat-map':
            fig = px.imshow(df_corr.corr())

        elif chart == 'pair':
            fig = px.scatter_matrix(df_pair, dimensions=feature, color="RainTomorrow")
        return fig

    return {}


@app.callback(
    dash.dependencies.Output('viz_location', 'figure'),
    [Input('button_location', 'n_clicks')],
    [State('dropdown_city_location', 'value'),
     State('dropdown_feature_location', 'value'),
     State('range_slider_year', 'value'),
     ]
)
def update_vis5(n_clicks, city, feature, year_range):
    if n_clicks:
        fig = None
        df_location = df[df['Location'].isin(city)]
        years = []
        for i in range(year_range[0], year_range[1] + 1):
            years.append(i)
        df_location = df_location[df_location['year'].isin(years)]

        df_location['year'] = df_location['year'].apply(str)
        df_location['year'] = df_location['year'].str[:-2]

        df_location = df_location.groupby(['Location', 'year'], as_index=False)[feature].mean()
        df_location = df_location.sort_values('year')
        fig = px.line(df_location, x="year", y=feature, color='Location')

        return fig

    return {}


@app.callback(
    dash.dependencies.Output('viz_directions', 'figure'),
    [Input('button_location', 'n_clicks')],
    [State('dropdown_city_location', 'value'),
     State('dropdown_feature_location', 'value'),
     State('range_slider_year', 'value'),
     ]
)
def update_vis6(n_clicks, cities, feature, year_range):
    if n_clicks:
        fig = None
        df_location = df[df['Location'].isin(cities)]
        years = []
        for i in range(year_range[0], year_range[1] + 1):
            years.append(i)
        df_location = df_location[df_location['year'].isin(years)]
        df_location['year'] = df_location['year'].apply(str)
        df_location['year'] = df_location['year'].str[:-2]
        cities_len = len(cities)
        cities_iter = (cities_len // 2 + 1) if (cities_len % 2 > 0) else (cities_len // 2)
        specs = []
        for i in range(0, cities_iter):
            specs.append([{'type': 'domain'}, {'type': 'domain'}])
        fig = make_subplots(rows=cities_iter, cols=2, specs=specs)

        row = 1
        col = 1
        for city in cities:
            df_city = df_location[df_location['Location'] == city]
            yes_count = len(df_city[df_location['RainTomorrow'] == 'Yes'].index)
            no_count = len(df_city[df_location['RainTomorrow'] == 'No'].index)
            fig.add_trace(go.Pie(labels=df_location['RainTomorrow'].unique(), values=[no_count, yes_count], name=city),
                          row, col)
            if (col % 2) == 0:
                row = row + 1
                col = 1
            else:
                col = col + 1
        fig.update_traces(hole=.4)
        fig.update_layout(
            title_text="City wise rainfall percentage"
        )
        return fig

    return {}


if __name__ == "__main__":
    app.run_server(debug=True)
