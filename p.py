from typing import Tuple
import json
from pathlib import Path
from plotly.subplots import make_subplots
import dash
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash.dependencies import Output, Input, State
from matplotlib.widgets import Button, Slider
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from project import *

import datetime
import sqlite3
import webbrowser
from time import sleep

from flask import Response

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Rainfall Prediction in Australia"
app.config.suppress_callback_exceptions=True
# server = app.server

# Reading Dataset

df = pd.read_csv('dataset/weatherAUS-processed.csv')
# aus_cities = json.load(open("dataset/australia-cities.geojson", "r"))
# aus_cities["name"] = aus_cities["properties"][""]
# aus_cities_id_map = {}
# for feature in aus_cities["features"]:
#     feature["Location"] = feature["properties"]["name"]

# print(aus_cities)


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dbc.Row([
        html.H1(children='Australian Rainfall - Visualization Dashboard'),
    ],
        justify="center",
        style={"margin-top": "0", "margin-bottom": "1px", "color": "#ffffff", "background-color": "#000000"}
    ),
    html.Div(id='page-content',style={"background-image": "url('https://www.basicplanet.com/wp-content/uploads/2017/01/Countries-with-Most-Rainfall-in-the-World.jpg')"})
])


index_page = dbc.Container([
    html.Title("Australian Rainfall Prediction"),
    # dbc.Row([
    #     html.H1(children='Australian Rainfall - Visualization Dashboard'),
    # ],
    #     justify="center",
    #     style={"margin-top": "0", "margin-bottom": "20px", "color": "#ffffff", "background-color": "#000000"}
    # ),
    dbc.Row([
        dbc.Col(
            dbc.Card(
                [
                    dbc.CardImg(src="http://media.heartlandtv.com/images/rain+graphic.jpg", top=True),
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
                    dbc.CardImg(src="http://media.heartlandtv.com/images/rain+graphic.jpg", top=True),
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
    ],style={"position":"relative","left":"110px"}),

    dbc.Row([
        dbc.Col(
            dbc.Card(
                [
                    dbc.CardImg(src="http://media.heartlandtv.com/images/rain+graphic.jpg", top=True),
                    dbc.CardBody(
                        [
                            html.H4("Rainfall Visualization", className="card-title"),
                            html.P(
                                "Did you know that we consider all the parameters to learn about past rainfall? Use these parameters as per your will to cluster similar rainfall!",
                                className="card-text",
                            ),
                            dbc.Button("Click to Visualize", color="warning"),
                        ]
                    ),
                ],
                style={"width": "18rem", "background-color": "#000000", "color": "#ffffff"},
            ),
        ),
        dbc.Col(
            dbc.Card(
                [
                    dbc.CardImg(src="http://media.heartlandtv.com/images/rain+graphic.jpg", top=True),
                    dbc.CardBody(
                        [
                            html.H4("Predict Rainfall", className="card-title"),
                            html.P(
                                "We provide you a ready made ML model and allow you to tweak thedemographics to predict future rainfall! The rainfall are visualized using a Gauge plot",
                                className="card-text",
                            ),
                            dbc.Button("Click to Visualize", color="warning"),
                        ]
                    ),
                ],
                style={"width": "18rem", "background-color": "#000000", "color": "#ffffff"},
            ),
        ),
    ],
        # justify="center,
        style={"position":"relative","left":"110px","margin-top": "20px"}
    ),
],
style={"margin-top": 0,"padding-top":"50px"}
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
            # justify="center",
            style={"margin-top": "50px", "margin-bottom": "20px", "color": "#ffffff", "background-color": "#000000", "padding-left": "1%"}
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
            # justify="center",
            style={"margin-top": "50px", "margin-bottom": "20px", "color": "#ffffff", "background-color": "#000000", "padding-left": "1%"}
        ),
        html.Div([
            dbc.FormGroup([
                dbc.Label("Select city"),
                dcc.Dropdown(id="dropdown_cities", value=1, options=fetch_cities()),

                # dbc.Label("Select Chart"),
                # dcc.Dropdown(id="dropdown_chart", value=2, options=[{"label": "Bar Chart", "value": "bar"},
                #                                                {"label": "line Chart", "value": "line"}]),
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
            # justify="center",
            style={"margin-top": "50px", "margin-bottom": "20px", "color": "#ffffff", "background-color": "#000000", "padding-left": "1%"}
        ),
        html.Div([
            dbc.FormGroup([
                # dbc.Label("Select city"),
                # dcc.Dropdown(id="dropdown_cities", value=1, options=fetch_cities()),

                dbc.Label("Select Chart"),
                dcc.RadioItems(id="dropdown_chart_corr",value=1, options=[{"label": "Heat-Map", "value": "heat-map"},
                                                                         {"label": "Pair Plot", "value": "pair"}],
                               labelStyle={'display': 'inline-block','padding-right':'15px'}),
                # dcc.Dropdown(id="dropdown_chart_corr", value=1, options=[{"label": "Heat-Map", "value": "heat-map"},
                #                                                          {"label": "Pair Plot", "value": "pair"}]),
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
    ], style={'font-style': 'italic', 'color':'red', 'background-color':'black'})
],style={"background-color":"#ffffff"})

page2 = dbc.Container([
    html.Div([
        dbc.Row([
            html.H3(children='Rainfall: At your Destination'),
        ],
            # justify="center",
            style={"margin-bottom": "20px", "color": "#ffffff", "background-color": "#000000", "padding-left": "1%"}
        ),
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
        dbc.Row([
            dbc.Col(dcc.Graph(id='viz_location')),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id='viz_directions')),
        ]),
    ], style={"padding-top": '20px' }),

], style={"background-color": "#ffffff"})

# Setting Web Layout
page = dbc.Container([
    # dbc.Row([
    #     html.H1(children='Australian Rainfall - Visualization Dashboard'),
    # ],
    #     justify="center",
    #     style={"margin-top": "50px", "margin-bottom": "20px", "color": "#ffffff", "background-color": "#000000"}
    # ),
    html.Div([
        dbc.FormGroup([
            dbc.Label("Select Year"),
            dcc.Dropdown(id="dropdown_years", value=1, options=fetch_years()),
            dbc.Label("Select feature"),
            dcc.Dropdown(id="dropdown_feature", value=2, options=fetch_numeric_columns()),
            html.Br()
        ]),
        dbc.Button('Show Map', id='button_map', color='success', style={'margin-bottom': '1em'},
                   block=True),
        dbc.Row([
            dbc.Col(dcc.Graph(id='viz1')),
        ]),
    ], style={"margin-top":"2px"}),
    dbc.Row([
            html.H3(children='Demogrphic Visualization'),
        ],
            # justify="center",
            style={"margin-top": "50px", "margin-bottom": "20px", "color": "#ffffff", "background-color": "#000000", "padding-left": "1%"}
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
        ]),
    dbc.Row([
            html.H3(children='Trend'),
        ],
            # justify="center",
            style={"margin-top": "50px", "margin-bottom": "20px", "color": "#ffffff", "background-color": "#000000", "padding-left": "1%"}
        ),
    html.Div([
            dbc.FormGroup([
                dbc.Label("Select city"),
                dcc.Dropdown(id="dropdown_cities", value=1, options=fetch_cities()),

                # dbc.Label("Select Chart"),
                # dcc.Dropdown(id="dropdown_chart", value=2, options=[{"label": "Bar Chart", "value": "bar"},
                #                                                {"label": "line Chart", "value": "line"}]),
                dbc.Label("Select feature"),
                dcc.Dropdown(id="dropdown_feature_trend", value=2, options=fetch_numeric_columns()),
                html.Br()
            ]),
            dbc.Button('Show Chart', id='button_trend', color='warning', style={'margin-bottom': '1em'},
                       block=True),
            dbc.Row([
                dbc.Col(dcc.Graph(id='viz3')),
            ]),
        ]),
    dbc.Row([
            html.H3(children='Correlation'),
        ],
            # justify="center",
            style={"margin-top": "50px", "margin-bottom": "20px", "color": "#ffffff", "background-color": "#000000", "padding-left": "1%"}
        ),
    html.Div([
            dbc.FormGroup([
                # dbc.Label("Select city"),
                # dcc.Dropdown(id="dropdown_cities", value=1, options=fetch_cities()),

                dbc.Label("Select Chart"),
                dcc.Dropdown(id="dropdown_chart_corr", value=1, options=[{"label": "Heat-Map", "value": "heat-map"},
                                                               {"label": "Pair Plot", "value": "pair"}]),
                dbc.Label("Select feature"),
                dcc.Dropdown(id="dropdown_feature_corr", value=2, options=fetch_numeric_columns(), multi=True),
                html.Br()
            ]),
            dbc.Button('Show Chart', id='button_corr', color='warning', style={'margin-bottom': '1em'},
                       block=True),
            dbc.Row([
                dbc.Col(dcc.Graph(id='viz4')),
            ]),
        ]),

],style={"background-color":"#ffffff"})

# Update the index
@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/rainfall-history':
        return page1
    elif pathname == '/location':
        return page2
    else:
        return index_page
    # You could also return a 404 "URL not found" page here



@app.callback(
    dash.dependencies.Output('viz1', 'figure'),
    [Input('button_map', 'n_clicks')],
    [State('dropdown_years', 'value'),
     State('dropdown_feature', 'value'),
     ]
)
def update_vis1(n_clicks, year, feature):
    if n_clicks:
        fig = None
        df_map = df.loc[df['year'] == year]
        df_map = df_map.groupby('Location', as_index=False)[feature].mean()

        # Creating Map
        # fig = px.choropleth(df, locations="Location", locationmode="country names", color="value",
        #                     hover_name="Location", color_continuous_scale=px.colors.sequential.Plasma)

        print(df_map)

        fig = px.choropleth(df_map, locations='Location',
                            color=feature,
                            # geojson=aus_cities,
                            color_continuous_scale="Viridis",
                            range_color=(0, 12),
                            # scope="asia",
                            labels={'feature': feature},
                            hover_data = ['MaxTemp'],
                            )

        fig.update_geos(fitbounds="locations", visible=True)
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
        elif chart =='line':
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
        # feature = fetch_numeric_columns()
        df_corr = df[feature]
        df_pair = df_corr.copy()
        df_pair['RainTomorrow'] = df['RainTomorrow']
        # print(df_pair)
        if chart == 'heat-map':
            fig = px.imshow(df_corr.corr())

        elif chart =='pair':
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
        for i in range(year_range[0], year_range[1]+1):
            print(i)
            years.append(i)
        df_location = df_location[df_location['year'].isin(years)]
        df_location['year'] = df_location['year'].apply(str)
        df_location['year'] = df_location['year'].str[:-2]

        df_location = df_location.groupby(['Location', 'year'], as_index=False)[feature].mean()
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
        print(year_range)
        years = []
        for i in range(year_range[0], year_range[1]+1):
            print(i)
            years.append(i)
        df_location = df_location[df_location['year'].isin(years)]
        df_location['year'] = df_location['year'].apply(str)
        df_location['year'] = df_location['year'].str[:-2]
        cities_len = len(cities)
        cities_iter = (cities_len//2 + 1) if (cities_len%2 > 0) else (cities_len//2)
        specs = []
        for i in range(0,cities_iter):
            specs.append([{'type':'domain'}, {'type':'domain'}])
        print("Specifications")
        print(specs)
        # specs=[[{'type':'domain'}, {'type':'domain'}], [{'type':'domain'}, {'type':'domain'}]]
        # df_location = df_location.groupby(['Location', 'year'], as_index=False)[feature].mean()
        fig = make_subplots(rows=cities_iter, cols=2, specs=specs)

        row = 1
        col = 1
        for city in cities:
            df_city = df_location[df_location['Location'] == city]
            yes_count = len(df_city[df_location['RainTomorrow'] == 'Yes'].index)
            no_count = len(df_city[df_location['RainTomorrow'] == 'No'].index)
            fig.add_trace(go.Pie(labels=df_location['RainTomorrow'].unique(), values=[no_count,yes_count],name=city),row, col)
            if (col % 2) == 0:
                row = row + 1
                col = 1
            else:
                col = col + 1
        fig.update_traces(hole=.4)
        fig.update_layout(
            title_text="City wise rainfall percentage"
            # annotations=[dict(text='AliceSprings', x=0.06, y=.8, font_size=10, showarrow=False),
            #              dict(text='Melbourne', x=0.7, y=0.8, font_size=10, showarrow=False),
            #              dict(text='Sydney', x=0.06, y=.187, font_size=10, showarrow=False),
            #              dict(text='Canberra', x=0.7, y=.187, font_size=10, showarrow=False)]
            )
        return fig

    return {}


if __name__ == "__main__":
    # app = dash_task()
    app.run_server(debug=True)
