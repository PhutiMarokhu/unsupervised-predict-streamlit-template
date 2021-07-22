"""

    Collaborative-based filtering for item recommendation.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: You are required to extend this baseline algorithm to enable more
    efficient and accurate computation of recommendations.

    !! You must not change the name and signature (arguments) of the
    prediction function, `collab_model` !!

    You must however change its contents (i.e. add your own collaborative
    filtering algorithm), as well as altering/adding any other functions
    as part of your improvement.

    ---------------------------------------------------------------------

    Description: Provided within this file is a baseline collaborative
    filtering algorithm for rating predictions on Movie data.

"""

# Script dependencies
import pandas as pd
import numpy as np

# Importing data
movies_df = pd.read_csv('resources/data/movies.csv',sep = ',',delimiter=',')
ratings_df = pd.read_csv('resources/data/ratings.csv')
ratings_df.drop(['timestamp'], axis=1,inplace=True)



# !! DO NOT CHANGE THIS FUNCTION SIGNATURE !!
# You are, however, encouraged to change its content.  
def collab_model(movie_list,top_n=10):
    """Performs Collaborative filtering based upon a list of movies supplied
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

    train_sample = ratings_df.merge(movies_df[['movieId', 'title']], on = 'movieId', how = 'left')
 
    train_pivot = train_sample.pivot_table(index = ['userId'], columns = ['title'], values = 'rating')
    
    

    ratings_1 = train_pivot[movie_list[0]]
    ratings_2 = train_pivot[movie_list[1]]
    ratings_3 = train_pivot[movie_list[2]]

    similar_movies_1 = train_pivot.corrwith(ratings_1).dropna().drop(movie_list[0]).sort_values(ascending = False)[2:11]
    similar_movies_2 = train_pivot.corrwith(ratings_2).dropna().drop(movie_list[1]).sort_values(ascending = False)[2:11]
    similar_movies_3 = train_pivot.corrwith(ratings_3).dropna().drop(movie_list[2]).sort_values(ascending = False)[2:11]

    listings = similar_movies_1.append(similar_movies_2).append(similar_movies_3)
    listings = listings.sort_values(ascending = False)[:top_n]
    
    
    return pd.Series(listings.index)
