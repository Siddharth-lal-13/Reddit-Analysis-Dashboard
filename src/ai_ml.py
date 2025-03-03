from google import genai
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd
import numpy as np
import os
# import tensorflow as tf

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


def generate_summary(text):
    """Generate a summary of a trend using the Gemini API."""
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=f"Summarize the following trend in simple terms: {text}",
    )
    return response.text


def get_topics(df, num_topics=3):
    """Perform topic modeling on post content."""
    vectorizer = CountVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(df['text'].fillna(''))

    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(X)

    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[-10:]]
        topics.append(f"Topic {topic_idx + 1}: {', '.join(top_words)}")
    return topics


def get_topic_distributions(df, num_topics=3):
    """Get topic distributions over time."""
    vectorizer = CountVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(df['text'].fillna(''))
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    topic_dist = lda.fit_transform(X)

    topic_df = pd.DataFrame(topic_dist, columns=[f'Topic {i + 1}' for i in range(num_topics)])
    topic_df['date'] = df['date'].values
    return topic_df.groupby('date').mean().reset_index()


def export_topic_embeddings(df, output_dir='/static'):
    pass
#     """Export topic embeddings for TensorFlow Projector using TensorFlow."""
#     # Ensure output directory exists
#     os.makedirs(output_dir, exist_ok=True)
#
#     # Generate topic distributions
#     vectorizer = CountVectorizer(stop_words='english', max_features=1000)
#     X = vectorizer.fit_transform(df['text'].fillna(''))
#     lda = LatentDirichletAllocation(n_components=3, random_state=42)
#     topic_dist = lda.fit_transform(X)
#
#     # Convert to TensorFlow tensor and reduce dimensions with TensorFlow
#     topic_tensor = tf.convert_to_tensor(topic_dist, dtype=tf.float32)
#     with tf.device('/CPU:0'):  # Ensure CPU usage for compatibility
#         # Simple PCA-like reduction using SVD in TensorFlow
#         _, _, v = tf.linalg.svd(topic_tensor)
#         embeddings_2d = tf.matmul(topic_tensor, v[:, :2]).numpy()  # Reduce to 2D
#
#     # Create metadata (post titles or IDs)
#     metadata = df['title'].fillna('Untitled').tolist()
#
#     # Save embeddings and metadata as TSV files
#     embeddings_path = os.path.join(output_dir, 'embeddings.tsv')
#     metadata_path = os.path.join(output_dir, 'metadata.tsv')
#     np.savetxt(embeddings_path, embeddings_2d, delimiter='\t')
#     with open(metadata_path, 'w', encoding='utf-8') as f:
#         f.write("Title\n")
#         for title in metadata:
#             f.write(f"{title}\n")
#
#     return embeddings_path, metadata_path
