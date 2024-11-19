import pandas as pd
import matplotlib.pyplot as plt
import numpy as np  # Add this line
from wordcloud import WordCloud 
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import gensim.downloader as api

# Read CSV
df = pd.read_csv(r"c:\Users\lenovo\Desktop\prodegy\twitter_training.csv", header=None)
df.columns = ["id", "topic", "sentiment", "text"]

# Clean the data
df = df.dropna()
df = df.drop_duplicates()

# Count sentiment frequency
sentiment_counts = df['sentiment'].value_counts()
print(sentiment_counts)

# Count sentiment frequency per topic
sentiment_frequency = df.groupby(['topic', 'sentiment']).size().reset_index(name='Frequency')

# Display the result
print(sentiment_frequency)

# Create a pivot table for plotting
pivot_table = sentiment_frequency.pivot(index='topic', columns='sentiment', values='Frequency').fillna(0)

# Plot the pivot table
pivot_table.plot(kind='bar', figsize=(14, 8))

# Improve the x-axis appearance
plt.title("Sentiment Frequency per Topic", fontsize=16)
plt.xlabel("Topic", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.xticks(rotation=30, ha='right', fontsize=10)  # Angled labels with right alignment
plt.tight_layout()  # Adjust layout to fit everything
plt.legend(title="Sentiment", fontsize=10)
plt.show()


# Load pre-trained GloVe embeddings
print("Loading pre-trained word embeddings...")
word_vectors = api.load("glove-wiki-gigaword-100")  # 100-dimensional GloVe embeddings

# Function to preprocess text
def preprocess_text(text):
    """
    Cleans text by removing special characters, digits, and extra spaces.
    Handles non-string inputs by converting them to an empty string.
    """
    if not isinstance(text, str):
        return ""  # Return an empty string for non-string inputs
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove non-alphabetic characters
    text = re.sub(r"\s+", " ", text)  # Replace multiple spaces with a single space
    return text.lower().strip()  # Convert to lowercase and strip leading/trailing spaces

# Function to compute the mean word embedding for a comment
def compute_embedding(comment):
    """
    Computes the mean embedding vector for a given comment.
    """
    words = comment.split()
    word_embeddings = [word_vectors[word] for word in words if word in word_vectors]
    if len(word_embeddings) > 0:
        return np.mean(word_embeddings, axis=0)  # Return mean of word embeddings
    else:
        # Return a zero vector if no valid words are found
        return np.zeros(word_vectors.vector_size)

# Load dataset
df = pd.read_csv(r"c:\Users\lenovo\Desktop\prodegy\twitter_training.csv", header=None)
df.columns = ["id", "topic", "sentiment", "text"]

# Preprocess text data
print("Preprocessing text data...")
df["clean_text"] = df["text"].apply(preprocess_text)

# Encode sentiment labels into numeric values
label_encoder = LabelEncoder()
df["sentiment_label"] = label_encoder.fit_transform(df["sentiment"])

# Compute word embeddings for each comment
print("Computing word embeddings for comments...")
embeddings = np.array([compute_embedding(comment) for comment in tqdm(df["clean_text"])])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    embeddings, df["sentiment_label"], test_size=0.2, random_state=42
)

# Train a logistic regression model
print("Training the classification model...")
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
