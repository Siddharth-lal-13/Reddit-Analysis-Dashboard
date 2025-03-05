import json
import pandas as pd
from datetime import datetime


def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    posts = []
    for post in data:
        post_data = post['data']
        posts.append({
            'id': post_data['id'],
            'title': post_data['title'],
            'selftext': post_data.get('selftext', ''),
            'subreddit': post_data['subreddit'],
            'created_utc': datetime.utcfromtimestamp(post_data['created_utc']),
            'ups': post_data['ups'],
            'num_comments': post_data['num_comments'],
            'author': post_data['author'],
            'url': post_data.get('url', '')
        })
    df = pd.DataFrame(posts)
    df['date'] = df['created_utc'].dt.date
    df['text'] = df['title'] + ' ' + df['selftext']
    return df


def filter_by_keyword(df, keyword):
    """Filter DataFrame by a keyword in title or selftext."""
    return df[df['text'].str.contains(keyword, case=False, na=False)]
