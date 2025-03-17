from src.data_processing import load_data
from src.visualizations import plot_time_series, plot_pie_chart, plot_network, plot_topic_time_series
from src.ai_ml import generate_summary, get_topics, export_topic_embeddings
import dash
from dash import dcc, html, Input, Output
import os
import pandas as pd

# Load data from project root
df = load_data(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'data.json'))

# Check for precomputed embeddings
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static')
embeddings_path = os.path.join(static_dir, 'embeddings.tsv')
metadata_path = os.path.join(static_dir, 'metadata.tsv')
if not (os.path.exists(embeddings_path) and os.path.exists(metadata_path)):
    try:
        export_topic_embeddings(df)  # This runs locally with TensorFlow
    except ImportError:
        print(
            "TensorFlow not installed. Please generate embeddings.tsv and metadata.tsv locally and upload to static/.")
        # embeddings_path, metadata_path = export_topic_embeddings(df)

# External stylesheet for styling
external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

# Create Dash app
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

# Initial visualizations
default_keyword = "anarchism"
fig1 = plot_time_series(df, keyword=default_keyword)
fig2 = plot_pie_chart(df)
fig3 = plot_network(df, keyword=default_keyword)
fig4 = plot_topic_time_series(df)


# Enhanced spike detection
def get_spike_info(dataframe, keyword=None):
    try:
        filtered_df = dataframe if not keyword else dataframe[
            dataframe['text'].str.contains(keyword, case=False, na=False)]
        if filtered_df.empty:
            return "No significant activity detected", "both subreddits", [], []
        if 'date' not in filtered_df.columns:
            if 'created_utc' in filtered_df.columns:
                filtered_df['date'] = pd.to_datetime(filtered_df['created_utc']).dt.date
            elif 'created' in filtered_df.columns:
                filtered_df['date'] = pd.to_datetime(filtered_df['created'], unit='s').dt.date
            else:
                return "No timestamp data available", "both subreddits", [], []
        daily_counts = filtered_df.groupby('date').size()
        if daily_counts.empty:
            return "No significant activity detected", "both subreddits", [], []
        spike_date = daily_counts.idxmax()
        spike_count = daily_counts.max()
        spike_day_df = filtered_df[filtered_df['date'] == spike_date]
        subreddit_counts = spike_day_df['subreddit'].value_counts()
        dominant_subreddit = subreddit_counts.idxmax() if not subreddit_counts.empty else "both subreddits"
        top_authors = spike_day_df['author'].value_counts().head(3).index.tolist()
        return f"peaked on {spike_date} with {spike_count} posts", dominant_subreddit, top_authors, spike_day_df
    except Exception as e:
        return f"Error analyzing trends: {str(e)}", "unknown", [], []


# Function to summarize topics into phrases (simple heuristic)
def summarize_topics(topics):
    topic_summaries = []
    used_summaries = set()  # Track used summaries to avoid duplicates
    for topic in topics[:3]:  # Top 3 topics
        words = topic.split(', ')[:5]  # First 5 words for brevity
        summary = None

        # Political/Governance Themes
        if any(w in words for w in ['government', 'trump', 'president', 'administration', 'state', 'federal']):
            summary = "Political Leadership and Governance"
        # Community Action/Protests
        elif any(w in words for w in ['protest', 'action', 'movement', 'rally']):
            summary = "Community Action and Protests"
        # Social Media/Content
        elif any(w in words for w in ['video', 'instagram', 'https', 'www', 'content', 'share']):
            summary = "Social Media and Content Sharing"
        # Ideological Discourse
        elif any(w in words for w in ['anarchism', 'conservative', 'ideology', 'theory', 'policy']):
            summary = "Ideological Perspectives and Debate"
        # Personal Expression/Opinion
        elif any(w in words for w in ['think', 'know', 'want', 'like', 'people', 'right']):
            summary = "Personal Opinions and Community Sentiment"
        # Default with contextual twist
        else:
            # Use the most frequent word as a hint
            top_word = words[0] if words else "Discussion"
            summary = f"Emerging {top_word.capitalize()} Trends"

        # Ensure uniqueness
        if summary in used_summaries:
            summary = f"Additional {summary.split(' and ')[0]} Insights" if ' and ' in summary else f"Broader {summary}"
        used_summaries.add(summary)
        topic_summaries.append(summary)

    return topic_summaries


# Initial setup
trend_info, dominant_subreddit, top_authors, spike_df = get_spike_info(df, default_keyword)
topics = get_topics(df)
topic_summaries = summarize_topics(topics)
trend_description = f"Posts containing '{default_keyword}' {trend_info} in {dominant_subreddit}. Top contributors: {', '.join(top_authors)}. Key topics: {', '.join(topic_summaries)}."
summary = generate_summary(trend_description)
narrative = f"On {trend_info.split('on ')[1].split(' with')[0]}, {dominant_subreddit} experienced a surge of {trend_info.split('with ')[1]} centered on '{default_keyword}'. Led by contributors such as {', '.join(top_authors)}, this activity highlighted themes like {', '.join(topic_summaries)}, indicating a potential reaction to a notable event or shift in community focus."

# Define layout with narrative and one-liners
app.layout = html.Div([
    html.H1("Reddit Analysis Dashboard", style={'textAlign': 'center', 'color': '#2c3e50'}),
    html.H2("Filter by Keyword", style={'textAlign': 'center'}),
    dcc.Dropdown(
        id='keyword-dropdown',
        options=[
            {'label': 'All Posts', 'value': ''},
            {'label': 'Anarchism', 'value': 'anarchism'},
            {'label': 'Conservative', 'value': 'conservative'},
            {'label': 'Protest', 'value': 'protest'},
            {'label': 'Trump', 'value': 'trump'},
            {'label': 'Policy', 'value': 'policy'}
        ],
        value=default_keyword,
        style={'width': '50%', 'margin': '0 auto'}
    ),
    html.H3("Insights at a Glance", style={'textAlign': 'center', 'color': '#2980b9', 'marginTop': '20px'}),
    html.P(id='narrative', children=narrative, style={'textAlign': 'center', 'fontSize': '16px', 'margin': '0 20px'}),
    html.H2(id='time-series-title', style={'textAlign': 'center'}),
    dcc.Graph(id='time-series-graph', figure=fig1),
    html.P("This temporal surge corresponds to shifts in community composition below.",
           style={'textAlign': 'center', 'fontSize': '14px', 'color': '#7f8c8d'}),
    html.H2("Post Distribution by Subreddit", style={'textAlign': 'center'}),
    dcc.Graph(figure=fig2),
    html.P("Dominant subreddits influenced the key contributors shown next.",
           style={'textAlign': 'center', 'fontSize': '14px', 'color': '#7f8c8d'}),
    html.H2(id='network-title', style={'textAlign': 'center'}),
    dcc.Graph(id='network-graph', figure=fig3),
    html.P("These interactions shaped the thematic trends that follow.",
           style={'textAlign': 'center', 'fontSize': '14px', 'color': '#7f8c8d'}),
    html.H2("Topic Prevalence Over Time", style={'textAlign': 'center'}),
    dcc.Graph(figure=fig4),
    html.H3("Trend Analysis Summary", style={'color': '#2980b9'}),
    html.P(id='summary-text', children=summary, style={'fontSize': '16px', 'margin': '0 20px'}),
    html.H3("Key Discussion Themes", style={'color': '#2980b9'}),
    html.Ul(id='topics-list', children=[html.Li(ts) for ts in topic_summaries]),  # Use summarized topics
    html.H3("Topic Embedding Visualization", style={'color': '#2980b9'}),
    html.P([
        "Explore topic embeddings in 2D using TensorFlow Projector: upload ",
        html.A("embeddings.tsv", href="/static/embeddings.tsv"),
        " and ",
        html.A("metadata.tsv", href="/static/metadata.tsv"),
        " to ",
        html.A("projector.tensorflow.org", href="https://projector.tensorflow.org", target="_blank")
    ])
], style={'padding': '20px', 'backgroundColor': '#ecf0f1'})


# Updated callback
@app.callback(
    [Output('time-series-graph', 'figure'),
     Output('network-graph', 'figure'),
     Output('summary-text', 'children'),
     Output('narrative', 'children'),
     Output('topics-list', 'children'),  # New output
     Output('time-series-title', 'children'),
     Output('network-title', 'children')],
    [Input('keyword-dropdown', 'value')]
)
def update_dashboard(keyword):
    keyword = keyword if keyword else None
    fig1 = plot_time_series(df, keyword=keyword)
    fig3 = plot_network(df, keyword=keyword)
    trend_info, dominant_subreddit, top_authors, spike_df = get_spike_info(df, keyword)
    topics = get_topics(df)
    topic_summaries = summarize_topics(topics)
    trend_description = (f"Posts containing '{keyword}' {trend_info} in {dominant_subreddit}. Top contributors: {', '.join(top_authors)}. Key topics: {', '.join(topic_summaries)}."
                         if keyword else f"Overall posting activity {trend_info} across {dominant_subreddit}. Top contributors: {', '.join(top_authors)}. Key topics: {', '.join(topic_summaries)}.")
    summary = generate_summary(trend_description)
    narrative = (f"On {trend_info.split('on ')[1].split(' with')[0]}, {dominant_subreddit} experienced a surge of {trend_info.split('with ')[1]} centered on '{keyword}'. "
                 f"Led by contributors such as {', '.join(top_authors)}, this activity highlighted themes like {', '.join(topic_summaries)}, "
                 f"indicating a potential reaction to a notable event or shift in community focus." if keyword else
                 f"Across the dataset, posting activity {trend_info}, with {dominant_subreddit} leading. Contributors like {', '.join(top_authors)} "
                 f"shaped discussions around {', '.join(topic_summaries)}, reflecting broad community engagement.")
    time_series_title = f"Posting Trends Over Time (Keyword: {keyword or 'All'})"
    network_title = f"Author-Subreddit Network (Keyword: {keyword or 'All'})"
    topics_list = [html.Li(ts) for ts in topic_summaries]
    return fig1, fig3, summary, narrative, topics_list, time_series_title, network_title


# Run the app
if __name__ == '__main__':
    app.run_server(debug=False)
