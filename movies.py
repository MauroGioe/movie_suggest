import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-bright')
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import process

ratings_data = pd.read_csv(r"ratings_small.csv")
ratings_data = ratings_data.drop('timestamp', axis = 1)
movie_names = pd.read_csv(r"movies_metadata.csv")
movie_names = movie_names[['title', 'genres']]


movie_names.info()

movie_data = pd.concat([ratings_data, movie_names], axis=1)
movie_data.head()



trend = pd.DataFrame(movie_data.groupby('title')['rating'].mean())
trend['total number of ratings'] = pd.DataFrame(movie_data.groupby('title')['rating'].count())



# Calculate mean rating of all movies and check the popular high rating movies
movie_data.groupby('title')['rating'].mean().sort_values(ascending=False).head(10)




movies_users = ratings_data.pivot(index=['userId'], columns=['movieId'], values='rating').fillna(0)
movies_users



from scipy.sparse import csr_matrix
mat_movies_users=csr_matrix(movies_users.values)
mat_movies_users


model_knn= NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)




model_knn.fit(mat_movies_users)


def Recommender(movie_name, data = mat_movies_users, model = model_knn, n_recommendations = 5):
    model.fit(data)
    movie_index = process.extractOne(movie_name, movie_names['title'])[2]
    print('Movie Selected: ', movie_names['title'][movie_index], ', Index: ', movie_index)
    print('Searching for recommendations.....')
    distances, indices = model.kneighbors(data[movie_index], n_neighbors=n_recommendations)
    recc_movie_indices = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())),
                                key=lambda x: x[1])[:0:-1]
    recommend_frame = []
    for val in recc_movie_indices:
        #         print(movie_names['title'][val[0]])
        recommend_frame.append({'Title': movie_names['title'][val[0]], 'Distance': val[1]})

    df = pd.DataFrame(recommend_frame, index=range(1, n_recommendations))
    df = df.iloc[::-1]
    print(df)
    return df
