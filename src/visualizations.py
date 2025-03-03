import pandas as pd
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
from src.data_processing import filter_by_keyword


def plot_time_series(df, keyword=None):
    """Create an interactive time series plot of posts over time by subreddit."""
    if keyword:
        df = filter_by_keyword(df, keyword)
    time_series = df.groupby(['date', 'subreddit']).size().reset_index(name='post_count')
    fig = px.line(
        time_series,
        x='date',
        y='post_count',
        color='subreddit',
        title=f'Posts Over Time by Subreddit (Keyword: {keyword or "All"})',
        labels={'post_count': 'Number of Posts', 'date': 'Date'}
    )
    return fig


def plot_topic_time_series(df):
    """Create a time series of topic prevalence."""
    from src.ai_ml import get_topic_distributions  # Adjusted to get distributions
    topic_dist = get_topic_distributions(df)
    fig = px.line(
        topic_dist,
        x='date',
        y=topic_dist.columns[1:],  # Exclude 'date'
        title='Topic Prevalence Over Time',
        labels={'value': 'Topic Weight', 'date': 'Date'}
    )
    return fig


def plot_pie_chart(df):
    """Create an interactive pie chart of post distribution by subreddit."""
    subreddit_counts = df['subreddit'].value_counts().reset_index()
    subreddit_counts.columns = ['subreddit', 'count']
    fig = px.pie(
        subreddit_counts,
        values='count',
        names='subreddit',
        title='Post Distribution by Subreddit'
    )
    return fig


def plot_network(df, keyword=None):
    """Create an interactive network graph of author-subreddit interactions."""
    if keyword:
        df = filter_by_keyword(df, keyword)
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_node(row['author'], type='author')
        G.add_node(row['subreddit'], type='subreddit')
        G.add_edge(row['author'], row['subreddit'], weight=row['ups'])

    pos = nx.spring_layout(G)
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=10,
            colorbar=dict(thickness=15, title='Connections', xanchor='left')
        )
    )

    node_text = [f"{node} ({G.nodes[node]['type']})" for node in G.nodes()]
    node_color = [len(list(G.neighbors(node))) for node in G.nodes()]
    node_trace.text = node_text
    node_trace.marker.color = node_color

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=f'Author-Subreddit Network (Keyword: {keyword or "All"})',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False)
        )
    )
    return fig
