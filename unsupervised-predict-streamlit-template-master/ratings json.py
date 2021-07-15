import json
import pandas as pd

# Importing ratings data
ratings_df = pd.read_csv('resources/data/ratings.csv')
ratings_df.drop(['timestamp'], axis=1,inplace=True)

ratings_list = list()
ratings_data = list()
# Iterate over the ratings data frame
for i in ratings_df.index:
	ratings_data.append((ratings_df.loc[i]['userId'], i, ratings_df.loc[i]['movieId'],
	ratings_df.loc[i]['rating']))
	ratings_list.append(ratings_data)

# Create JSON files for data and movie titles for faster load to the Recommmender
ratings_dump = r'resources/data/ratings.json'
with open(ratings_dump, 'w+', encoding='utf-8') as f:
    json.dump(ratings_list, f)