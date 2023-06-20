import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
# latent Dirichlet allocation
basepath = '/mnt/d/Books/Python/ml-with-pytorch-scikit/aclImdb'
csv_path = basepath + "/movie_data.csv"
df = pd.read_csv(csv_path, encoding='utf-8')

df = df.rename(columns={"0": "review", "1": "sentiment"})

count = CountVectorizer(stop_words='english', max_df=.1, max_features=5000)
X = count.fit_transform(df['review'].values)
print(X)
lda = LatentDirichletAllocation(
    n_components=10, random_state=123, learning_method='batch')
X_topics = lda.fit_transform(X)

print(lda.components_.shape)
n_top_words = 5
feature_names = count.get_feature_names_out()
for topic_idx, topic in enumerate(lda.components_):
    print(f'Topic {(topic_idx + 1)}:')
    print(' '.join([feature_names[i]
          for i in topic.argsort()[:-n_top_words - 1: -1]]))
