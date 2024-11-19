import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud 

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