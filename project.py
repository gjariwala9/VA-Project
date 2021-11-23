import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from typing import List, Dict
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider


def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    #print(df.select_dtypes(include='number'))
    return list(df.select_dtypes(include='number').columns.values)

def get_text_categorical_columns(df: pd.DataFrame) -> List[str]:
    #print(df.select_dtypes(include='object'))
    return list(df.select_dtypes(include='object').columns.values)

def fix_outliers(df: pd.DataFrame, column: str) -> pd.DataFrame:
    numeric_cols = get_numeric_columns(df)
    if(column in numeric_cols):
        temp_df = df.copy()
        #Using mean - standard deviation rule to find the lowest and highest limit
        highest_limit = temp_df[column].mean() + 3*temp_df[column].std()
        lowest_limit = temp_df[column].mean() - 3*temp_df[column].std()
        #Replace values less than lower limit to lower limit and values higher than upper limit to upper limit value
        temp_df[column] = np.where(temp_df[column] > highest_limit,highest_limit,np.where(temp_df[column] < lowest_limit,lowest_limit,temp_df[column]))
        return temp_df
    return df


def normalize_column(df_column: pd.Series) -> pd.Series:

    if df_column.dtype in [np.int64, np.float64]:
        df_col_min = df_column.min()
        df_col_max = df_column.max()
        # Calculating min - max normalization
        norm_col = (df_column - df_col_min) / (df_col_max - df_col_min)
        return norm_col
    return df_column

def generate_one_hot_encoder(df_column: pd.Series) -> OneHotEncoder:
    oneHotEncoder = OneHotEncoder()
    return oneHotEncoder.fit(df_column.values.reshape(-1, 1))

def replace_with_one_hot_encoder(df: pd.DataFrame, column: str, ohe: OneHotEncoder, ohe_column_names: List[str]) -> pd.DataFrame:
    df_copy = df.copy()
    encoded_df = pd.DataFrame(ohe.transform(df_copy[[column]]).toarray())
    encoded_df.columns = ohe_column_names
    df_copy = df_copy.join(encoded_df)
    df_copy.drop(column, inplace=True, axis=1)
    return df_copy

def replace_one_hot_encoder_with_original_column(df: pd.DataFrame,
                                                 columns: List[str],
                                                 ohe: OneHotEncoder,
                                                 original_column_name: str) -> pd.DataFrame:
    df_copy = df.copy()
    df_one_hot_encoder = df_copy[columns]
    df_copy[original_column_name] = ohe.inverse_transform(df_one_hot_encoder.values)
    df_copy.drop(columns, axis=1, inplace=True)
    return df_copy

def simple_k_means(x: pd.DataFrame, n_clusters=3, score_metric='euclidean') -> Dict:
    model = KMeans(n_clusters=n_clusters)
    clusters = model.fit_transform(x)

    # There are many methods of deciding a score of a cluster model. Here is one example:
    score = metrics.silhouette_score(x, model.labels_, metric=score_metric)
    return dict(model=model, score=score, clusters=clusters)

def plotly_bar_chart(df: pd.DataFrame):
    # Creating bar chart
    fig = px.bar(df, x='x', y='y')
    return fig


def plotly_pie_chart(df: pd.DataFrame):

    # creating pie chart
    fig = px.pie(df, names='x', values='x')
    return fig


def plotly_histogram(df: pd.DataFrame, n_bins: int):
    # Creating histogram with n number of bins
    fig = px.histogram(df, x='x', nbins=n_bins)
    return fig


def plotly_polar_chart(df: pd.DataFrame):
    # Calculating angles
    df['a'] = 2 * np.pi * df['y']

    # Creating polar chart
    fig = px.line_polar(df, r='x', theta='a')
    return fig


def plotly_heatmap_chart(df: pd.DataFrame):
    # creating heatmap
    fig = px.imshow(df)
    return fig


def plotly_table(df: pd.DataFrame):
    data = [df[i] for i in list(df.columns)]
    # inserting index values to data
    data.insert(0, df.index.to_list())

    # Creating table
    fig = go.Figure(data=[go.Table(header=dict(values=["index"] + list(df.columns)),
                                   cells=dict(values=data))])
    return fig


def plotly_contour_chart(df: pd.DataFrame):

    # Creating contour chart
    fig = go.Figure(data=
    go.Contour(
        z=df
    ))
    return fig


def plotly_composite_line_bar(df: pd.DataFrame):

    # Creating Bar Chart
    fig = plotly_bar_chart(df)

    # Adding line chart on bar chart
    fig.add_trace(
        go.Scatter(x=df['x'], y=df['y'])
    )
    return fig


def plotly_subgraphs(df: pd.DataFrame):
    # Creating structure of graph
    fig = make_subplots(rows=2, cols=2, start_cell="top-left")

    # Adding line chart to main graph
    fig.add_trace(go.Scatter(x=df['x1'], y=df['y1']), row=1, col=1)

    # Adding scatter plot to main graph
    fig.add_trace(go.Scatter(x=df['x2'], y=df['y2'], mode="markers"), row=1, col=2)

    # Adding Bar Chart to main graph
    fig.add_trace(go.Bar(x=df['x3']), row=2, col=1)

    # Adding Stacked Area Plot to main graph
    trace1 = go.Scatter(x=df['x3'], y=df['y3'], stackgroup='one')
    trace2 = go.Scatter(x=df['x4'], y=df['y4'], stackgroup='one')
    fig.add_traces([trace1, trace2], rows=2, cols=2)

    return fig

def matplotlib_cluster_interactivity():
    df = pd.read_csv('dataset/weatherAUS-processed.csv')

    # Dropping Species column from the dataset
    df = df.drop('species', axis=1)

    # Setting default clusters to 2
    model_cluster = simple_k_means(df, n_clusters=2)

    # Creating scatter plot
    fig, ax = plt.subplots()
    ax.scatter(x=df['petal_width'], y=df['petal_length'], c=model_cluster['model'].labels_, cmap='gist_rainbow')
    plt.subplots_adjust(bottom=0.2)

    class Index(object):

        # Function to update graph on change in cluster number
        def graph_update(self, value):
            model_cluster = simple_k_means(df, n_clusters=value)
            ax.clear()
            ax.scatter(x=df['petal_width'], y=df['petal_length'], c=model_cluster['model'].labels_, cmap='gist_rainbow')
            plt.draw()

    callback = Index()

    # Setting slider coordinates on graph
    graph_slider = plt.axes([0.35, 0.1, 0.35, 0.03])

    # Creating Slider
    slider = Slider(graph_slider, 'n_clusters', 2, 10, valstep=1)

    # Calling callback function on change in slider value
    slider.on_changed(callback.graph_update)

    return fig