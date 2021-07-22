"""

    Content-based filtering for item recommendation.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: You are required to extend this baseline algorithm to enable more
    efficient and accurate computation of recommendations.

    !! You must not change the name and signature (arguments) of the
    prediction function, `content_model` !!

    You must however change its contents (i.e. add your own content-based
    filtering algorithm), as well as altering/adding any other functions
    as part of your improvement.

    ---------------------------------------------------------------------

    Description: Provided within this file is a baseline content-based
    filtering algorithm for rating predictions on Movie data.

"""

# Script dependencies
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Importing data
df_movies = pd.read_csv('resources/data/movies.csv', sep = ',',delimiter=',')
df_imdb = pd.read_csv('resources/data/imdb_data.csv')
df_movies.dropna(inplace=True)

def data_preprocessing(subset_size):
    """Prepare data for use within Content filtering algorithm.

    Parameters
    ----------
    subset_size : int
        Number of movies to use within the algorithm.

    Returns
    -------
    Pandas Dataframe
        Subset of movies selected for content-based filtering.

    """

    # Merge the movies and imdb data sets
    # Don't remove rows containing Nan values, this will remove too much data
    movies_content = df_movies[
        ['movieId', 'title', 'genres']
    ].merge(df_imdb[
        ['movieId', 'title_cast', 'director', 'plot_keywords']
    ], on = 'movieId', how = 'left').fillna('')

    # Remove all rows contain no genres listed
    # Data Frame indices needs to be reset to correpsond to order
    movies_content = movies_content[movies_content['genres'] != '(no genres listed)'].reset_index()

    # Concatenate genres, plot keywords, actors and director columns with a vertical line to 
    # to maintain a consistent seperator
    movies_content['genres_plot_cast_director'] = movies_content[
        ['genres', 'plot_keywords', 'title_cast', 'director']
    ].agg('|'.join, axis = 1)

    movies_subset = movies_content[:subset_size]
    return movies_subset

# !! DO NOT CHANGE THIS FUNCTION SIGNATURE !!
# You are, however, encouraged to change its content.  
def content_model(movie_list,top_n=10):
    """Performs Content filtering based upon a list of movies supplied
       by the app user.

    Parameters
    ----------
    movie_list : list (str)
        Favorite movies chosen by the app user.
    top_n : type
        Number of top recommendations to return to the user.

    Returns
    -------
    list (str)
        Titles of the top-n movie recommendations to the user.

    """
    # Initializing the empty list of recommended movies
    recommended_movies = []
    data = data_preprocessing(27000)
    # Instantiating and generating the count matrix
    tfidf_vec = TfidfVectorizer(analyzer = lambda x: x.split('|'))
    tfidf_matrix = tfidf_vec.fit_transform(data['genres_plot_cast_director'])
    indices = pd.Series(data.index, index = data['title'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    # Getting the index of the movie that matches the title
    idx_1 = indices[movie_list[0]]
    idx_2 = indices[movie_list[1]]
    idx_3 = indices[movie_list[2]]
    # Creating a list of tuples containing the movie index and similarity score
    rank_1 = list(enumerate(cosine_sim[idx_1]))
    rank_2 = list(enumerate(cosine_sim[idx_2]))
    rank_3 = list(enumerate(cosine_sim[idx_3]))
    # Sorting each list by similarity score and only returning half of the total number of recommendations
    score_series_1 = sorted(rank_1, key = lambda x: x[1], reverse = True)[:int(top_n / 2)]
    score_series_2 = sorted(rank_2, key = lambda x: x[1], reverse = True)[:int(top_n / 2)]
    score_series_3 = sorted(rank_3, key = lambda x: x[1], reverse = True)[:int(top_n / 2)]
    # Concatenate lists 
    listings = score_series_1 + score_series_2 + score_series_3
    # Iterate through list to make sure the chosen movies are excluded
    listings = [x for x in listings if x[0] not in [idx_1, idx_2, idx_3]]
    # Sorting the final recommendation list
    listings = sorted(listings, key = lambda x: x[1], reverse = True)

    # Get top 10 recommendations
    movie_indices = [i[0] for i in listings[:top_n]]
    recommended_movies = data['title'].iloc[movie_indices].values
    return recommended_movies