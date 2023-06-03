from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

genre_cols = [
    "genre_unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]
movies_cols = [
    'movie_id', 'title', 'release_date', "video_release_date", "imdb_url"
] + genre_cols

df_movies = pd.read_csv('/mnt/d/Books/Python/ml-with-pytorch-scikit/excersize-movie-recommendation/ml-100k/u.item',
                        sep='|', names=movies_cols, encoding='latin-1')


users_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
df_users = pd.read_csv(
    '/mnt/d/Books/Python/ml-with-pytorch-scikit/excersize-movie-recommendation/ml-100k/u.user', sep='|', names=users_cols, encoding='latin-1')

ratings_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
df_ratings = pd.read_csv(
    '/mnt/d/Books/Python/ml-with-pytorch-scikit/excersize-movie-recommendation/ml-100k/u.data', sep='\t', names=ratings_cols, encoding='latin-1')

print(df_movies.head())
print('-' * 10)
print(df_users.head())
print('-' * 10)
print(df_ratings.head())
print('-' * 10)

df_users[['age']].hist()
# plt.show()
df_users[['sex', 'occupation']].value_counts().plot(kind='bar')
# plt.show()
df_ratings[['rating']].hist()
# plt.show()

# figure out the movie ratings
df_movies_ratings = df_movies.merge(
    df_ratings
    .groupby('movie_id', as_index=False)
    .agg({'rating': ['count', 'mean']}),
    on='movie_id')

print(df_movies_ratings.head())
print(df_movies_ratings.columns.values)
df_movies_ratings = df_movies_ratings.rename(
    columns={('rating', 'count'): 'rating count', ('rating', 'mean'): 'rating mean'})

print('Top 10 most rated movies: ')
print(df_movies_ratings[['title', 'rating count', 'rating mean']].sort_values(
    'rating count', ascending=False).head(10))

print('Top 10 highest rated movies with atleast 20 ratings: ')
print(df_movies_ratings[['title', 'rating count', 'rating mean']
                        ].loc[df_movies_ratings['rating count'] > 20].sort_values('rating mean', ascending=False).head(10))
