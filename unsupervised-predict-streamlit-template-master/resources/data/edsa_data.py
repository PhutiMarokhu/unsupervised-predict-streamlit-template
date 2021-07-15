import json
import pandas as pd


# Load dataset to a data frame
data = pd.read_csv('resources/data/movies.csv', sep = ',',delimiter=',')
# Create a new data frame with relevant columns only
df = data[['movieId', 'title', 'genres']].copy()
# Fetch genres of all movies
genres_all_movies = [df.loc[i]['genres'].split('|') for i in df.index]
# Find the list of genres of all movies in alphabetical order
genres = sorted(list(set([item for sublist in genres_all_movies for item in sublist])))

# Importing ratings data
ratings_df = pd.read_csv('resources/data/ratings.csv')
ratings_df.drop(['timestamp'], axis=1,inplace=True)


# Initialize lists for movie data as well as titles
full_data = list()
movie_titles = list()
# Iterate over the data frame
for i in df.index:
    # Append movie title and the index of the movie
    movie_titles.append((df.loc[i]['title'].strip(), i))
    # Add list of genres of the movies (1/0) to movie data
    movie_data = [1 if genre in df.loc[i]['genres'].split('|') else 0 for genre in genres]
    # Add record of movie to main data list
    full_data.append(movie_data)

ratings_list = list()
ratings_data = list()
# Iterate over the ratings data frame
for i in ratings_df.index:
	ratings_data.append((ratings_df.loc[i]['userId'], i, ratings_df.loc[i]['movieId'],
	ratings_df.loc[i]['rating']))
	ratings_list.append(ratings_data)


# Create JSON files for data and movie titles for faster load to the Recommmender
data_dump = r'resources/data/data.json'
titles_dump = r'resources/data/titles.json'
ratings_dump = r'resources/data/ratings.json'
#with open(data_dump, 'w+', encoding='utf-8') as f:
##   json.dump(full_data, f)
#with open(titles_dump, 'w+', encoding='utf-8') as f:
#    json.dump(movie_titles, f)
with open(ratings_dump, 'w+', encoding='utf-8') as f:
    json.dump(ratings_list, f)