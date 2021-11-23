from typing import Tuple
import json
from pathlib import Path

import dash
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
app.config.suppress_callback_exceptions=True
# server = app.server

# Reading Dataset

df = pd.read_csv('dataset/weatherAUS-processed.csv')
aus_cities = json.load(open("dataset/australia.cities.geo.json", "r"))
# aus_cities["name"] = aus_cities["properties"][""]
# aus_cities_id_map = {}
for feature in aus_cities["features"]:
    feature["Location"] = feature["properties"]["name"]

# print(aus_cities)

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


index_page = dbc.Container([
    html.Title("Australian Rainfall Prediction"),
    dbc.Row([
        html.H1(children='Australian Rainfall - Visualization Dashboard'),
    ],
        justify="center",
        style={"margin-top": "50px", "margin-bottom": "20px", "color": "#ffffff", "background-color": "#000000"}
    ),
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
                            dbc.Button("Click to Visualize", color="warning"),
                        ]
                    ),
                ],
                style={"width": "18rem", "height": "23rem", "background-color": "#000000", "color": "#ffffff"},
            ),
        )
    ],
    ),

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
        # justify="center",
        style={"margin-top": "20px"}
    ),
])


def fetch_years():
    lst = df['year'].unique()
    lst = lst.tolist()
    years = []
    for year in lst:
        years.append({"label": year, "value": year})
    return years


def fetch_numeric_columns():
    num_cols = []
    numeric_cols = get_numeric_columns(df)
    for col in numeric_cols:
        num_cols.append({"label": col, "value": col})
    return num_cols


# Setting Web Layout
page1 = dbc.Container([
    dbc.Row([
        html.H1(children='Australian Rainfall - Visualization Dashboard'),
    ],
        justify="center",
        style={"margin-top": "50px", "margin-bottom": "20px", "color": "#ffffff", "background-color": "#000000"}
    ),
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
    ]),

])

# Update the index
@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/rainfall-history':
        return page1
    # elif pathname == '/page-2':
    #     return page_2_layout
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
                            geojson=aus_cities,
                            color_continuous_scale="Viridis",
                            range_color=(0, 12),
                            # scope="asia",
                            labels={'feature': feature},
                            hover_data = ['MaxTemp'],
                            )

        fig.update_geos(fitbounds="locations", visible=True)
        return fig

    return {}

if __name__ == "__main__":
    # app = dash_task()
    app.run_server(debug=True)
