"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
st.set_page_config(
    page_title = "AM4 Stramlit Movie Recommender",
    page_icon="🎥"
    )

# Data handling dependencies
import pandas as pd
import numpy as np

# Packages for visualization
from plotly.offline import init_notebook_mode
import plotly.graph_objects as go

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model

#Github links to data
genome_link = "https://github.com/PhutiMarokhu/unsupervised-predict-streamlit-template/blob/a2e55a282129cf259830e907f4fb9d1f6e99b4fb/Snipped%20of%20datasets(100)/genome_scores_df.csv"
genome_tag_link = "https://github.com/PhutiMarokhu/unsupervised-predict-streamlit-template/blob/a2e55a282129cf259830e907f4fb9d1f6e99b4fb/Snipped%20of%20datasets(100)/genome_scores_df.csv"
imdb_link = "https://github.com/PhutiMarokhu/unsupervised-predict-streamlit-template/blob/a2e55a282129cf259830e907f4fb9d1f6e99b4fb/Snipped%20of%20datasets(100)/imdb_df.csv"
links_link = "https://github.com/PhutiMarokhu/unsupervised-predict-streamlit-template/blob/a2e55a282129cf259830e907f4fb9d1f6e99b4fb/Snipped%20of%20datasets(100)/links_df.csv"
movies_link = "https://github.com/PhutiMarokhu/unsupervised-predict-streamlit-template/blob/a2e55a282129cf259830e907f4fb9d1f6e99b4fb/Snipped%20of%20datasets(100)/movies_df.csv"
tags_link = "https://github.com/PhutiMarokhu/unsupervised-predict-streamlit-template/blob/a2e55a282129cf259830e907f4fb9d1f6e99b4fb/Snipped%20of%20datasets(100)/tags_df.csv"
test_link = "https://github.com/PhutiMarokhu/unsupervised-predict-streamlit-template/blob/a2e55a282129cf259830e907f4fb9d1f6e99b4fb/Snipped%20of%20datasets(100)/test_df.csv"
train_link = "https://github.com/PhutiMarokhu/unsupervised-predict-streamlit-template/blob/a2e55a282129cf259830e907f4fb9d1f6e99b4fb/Snipped%20of%20datasets(100)/train_df.csv"

# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')

movies_data = movies = pd.read_csv('resources/data/movies_df.csv', sep = ',',delimiter=',')
genome_data = pd.read_csv('resources/data/genome_scores_df.csv', sep = ',',delimiter=',')
genome_tags_data = pd.read_csv('resources/data/genome_tags_df.csv', sep = ',',delimiter=',')
links_data = pd.read_csv('resources/data/links_df.csv', sep = ',',delimiter=',')
train_data = pd.read_csv('resources/data/train_df.csv', sep = ',',delimiter=',')
test_data = pd.read_csv('resources/data/test_df.csv', sep = ',',delimiter=',')
imdb_data = pd.read_csv('resources/data/imdb_df.csv', sep = ',',delimiter=',')
tags_data = pd.read_csv('resources/data/tags_df.csv', sep = ',',delimiter=',')

# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Recommender System","Collected Data","Preprocessing methods", "EDA", "Solution Overview"]

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "Recommender System":
        # Header contents
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Movie posters.jpeg', width =800)
        ##st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('Fisrt Option',title_list[14930:15200])
        movie_2 = st.selectbox('Second Option',title_list[25055:25255])
        movie_3 = st.selectbox('Third Option',title_list[21100:21200])
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")

                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                           top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                        
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    if page_selection == "Collected Data":
        st.title("Collected data and descriptions")
        st.subheader("We imported 8 databases for our model training and fitting. Namely:")

        counter = st.slider("Select number of values to view from dataframes", 5, 20, 5)

        st.write("""*1. Movies Dataset*  \n Database containing the informaion regarding the movies 
        such as movieId, title, genres.""")
        st.text("Observation: \n\t •Contains no null values \n\t •Has 62423 rows and 3 columns")
        st.dataframe(movies_data.head(counter))
        st.markdown(f"[Get movies data]({movies_link})")

        st.write("""*2. IMDB Dataset*  \n Database containging info scraped from imdb like title_cast, 
        director, runtime, budget, etc.""")
        st.text("""Observation: \n\t •Contains null values in the title_cast, director, runtime, budget 
        and\n\t  plot_keywords columns \n\t •Has 27278 rows and 6 columns""")
        st.dataframe(imdb_data.head(counter))
        st.markdown(f"[Get IMDB data]({imdb_link})")

        st.write("""*3. Genome Scores Dataset*  \n Each genome tag score reflects the relevance of a tag 
        to a movie on a scale of 0 to 1. They were computed us- ing a machine learning algorithm based 
        on user-contributed content that includes tags, ratings and textual reviews.""")
        st.text("""Observation: \n\t •Contains no null values \n\t •Has 15584448 rows and 3 columns \n\t 
        •The info() method wasn't any help in displaying null values therefore we used \n\t  the isnull() 
        and sum() methods to see if each columns contains null values""")
        st.dataframe(genome_data.head(counter))
        st.markdown(f"[Get genome scores data]({genome_link})")

        st.write("""*4. Genome Tags Dataset*  \n  The tag genome representation is built from user input 
        on the website, where users can apply tags to movies and rate them in order to provide the information
         needed to make recommendations.""")
        st.text("Observation: \n\t •Contains no null values \n\t •Has 1128 rows and 2 columns")
        st.dataframe(genome_tags_data.head(counter))
        st.markdown(f"[Get genome tag data]({genome_tag_link})")

        st.write("""*5. Train Dataset*  \n Contains userId, movieId, ratings and timestamp of movies. These are ratings 
        given by users for movies as well as the date and time these ratings were given.""")
        st.text("""Observation: \n\t •Contains no null values \n\t •Has 10000038 rows and 4 columns \n\t 
        •The info() method wasn't any help in displaying null values therefore we used \n\t  the isnull() and 
        sum() methods to see if each columns contains null values""")
        st.dataframe(train_data.head(counter))
        st.markdown(f"[Get train data]({train_link})")
        
        st.write("*6. Test Dataset*  \n ")
        st.text("""Observation: \n\t •Contains no null values \n\t •Has 5000019 rows and 2 columns \n\t 
        •The info() method wasn't any help in displaying null values therefore we used \n\t  the isnull() 
        and sum() methods to see if each columns contains null values""")
        st.dataframe(test_data.head(counter))
        st.markdown(f"[Get test data]({test_link})")
        
        st.write("""*7. Tags Dataset*  \n A tag is a keyword or term assigned to a piece of information. 
        This kind of metadata helps describe an item and allows it to be found again by browsing or searching. 
        These are added by users as extra information that may not come with the movie. """)
        st.text("Observation: \n\t •Contains null values in the tag column \n\t •Has 1093360 rows and 4 columns")
        st.dataframe(tags_data.head(counter))
        st.markdown(f"[Get tags data]({tags_link})")

        
        st.write("""*8. Links Dataset*  \n Database containg the IMDB_id and TMDB_id which can be compiled to 
        get the full link to the respective websites.""")
        st.text("Observation: \n\t •Contains null values in the tmdbID column\n\t •Has 62423 rows and 3 columns")
        st.dataframe(links_data.head(counter))
        st.markdown(f"[Get links data]({links_link})")

    if page_selection == "Preprocessing methods":
        st.title("Data Preprocessing methods")
        st.markdown('Memory reduction')
        st.write("""Memory reduction can be performed by changing each column's daya type, 
        to a data type that is best suited for the range of values it contains.  
        \n**_Why is memory redcution important?_**
        \nChoosing the right data types for your tables, stored procedures, and variables not 
        only improves performance by ensuring a correct execution plan, but it also improves 
        data integrity by ensuring that the correct data is stored within a database.""")
        st.write("""Movies Data Set: movieId --> uint32

IMDB Data Set: movieId --> uint32 | runtime --> float16

Train Data Set: userId --> uint32 | movieId --> uint32 | rating --> float16 | timestamp --> uint32

Test Data Set: userId --> uint32 | movieId --> uint32

Links Data Set: movieId --> uint | imdbId --> uint32 | tmdbId --> will be removed \n""")
        st.image("resources/imgs/Memory usage.png", caption = "Memory(GB) saved by choosing the correct data type",
            use_column_width=True )
        st.markdown("""Just by correctly assigning the data types, we have free up over 200Mb of storage 
        which is a lot of space, especially for a dataset this small incomparison to those of major tech company's 
        which are likely in the terabytes. And in a profit sensitive market like this one, storage estate is at a 
        premium. The combined memory usage has been reduced by approximately 221MB. This will make the transfer of 
        data much faster and will reduce the amount of resources needed to process the data.

        \n\nIf the data frames are merged then it will increase the memory usage because of the increase in dimensions/columns.
        To avoid increasing the amount of resources being used, remove variables that you are no longer using by using del 
        <variable_name>. For example, if you merge the data frames and store it in a new variable then there is no need to 
        keep the individual data frame variables""")
        


    if page_selection == "EDA":
        st.title("Data Exploration")
        st.write("""There are multiple datasets containing information that we don't know if 
        it is relevant or not. We need to explore our data to get a better understanding of it's context.""")
        st.header("Summary of our data")
        st.image("resources/imgs/movie database sum.png")
        st.write("""We are working with a total of just over 62,000 movies, whose average duration is 100.3
        minutes-roughly 1:41 minutes. Some research suggests that our attention span has been decreasing over 
        the decades, with an average attention span of 12 seconds in 2000 to 8 seconds now. Some movie critics are 
        wondering if movie runtimes should also decrease with the decades to keep the audience engaged. 
        \nOur data was collected from [MovieLens.org](https://movielens.org/), a research site run 
        by GroupLens Research at the University of Minnesota. Our database consists of 162,500 users with a combined 
        avergae ratings of 3.53. The median of these ratings is 2.5(out of 5). So the fact that the average ratings 
        in the database is 1 unit higher than the median tells us that on avergae, movies are well made or 
        rather, movies are generally well received and rated by the public.""")

        st.header("Movie Genres composition")
        st.image("resources/imgs/Most Genre.png")
        st.write("""The majority of the the genre of movies that are being watched/produced are Drama. Making up 23% 
        of all the movies in our databases. This is possibly because drama can occur adjacent to other genres for the same movie 
        because most movies will contain a good degree of drama to it. Be it Action, Comedy and even Animation. There are also a 
        lot of movies which are added/published without any genre being
         """)

        st.header("Rating Distribution")
        st.image("resources/imgs/Ratings distri.png")
        st.write("""We will be looking at how ratings are distributed in df_train DataFrame.
        26.5% of all ratings is 4 which is the majority.
        The distribution shows a skewness in the positive direction""")

        st.header("Number of Ratings Per Year ")
        st.image("resources/imgs/Number of Ratings per year.png")
        st.write("""Knowing that there are so many users, we just can't look at them all .Therefore we will visualise the first 50 users and the total amount of rating for each of them 
        It appears that there are users who have rated only a few movies, 
        this implies that not all users are equivalent to suggest movie recommendations to other users.
        This is only one perspective, we could also view it as movies that only received one or very few ratings. This can be caused by a lack of popularity or received a bad rating by the first user then became overlooked by other users.Based on the distribution above, it seems that there are users with next to nothing ratings. This is an indication that there are users whose ratings could not contribute to recommending movies.    
              """)
        
        st.header("Number of ratings per user")
        st.image("resources/imgs/Number of ratings per user.png")
        st.write(""" We will be looking at the top 10 movies with the most total ratings (this is only the total number of ratings given, not the average ratings)
        The movie with the most ratings is Shawshank Redemption,
        The (1994), the movie is about a banker who is convicted for the murder of his wife and her lover and is sentenced to two consecutive life sentences at the Shawshank State Prison.
        This movie is based on Rita Hayworth and Shawshank Redemption by Stephen King and it is claimed to be amongs the best movies ever made in World Cinema and applauded by many film critics.
        Forrest Gump (1994) is about a man with a low IQ, recounts the early years of his life when he found himself in the middle of key historical events. This movie has been voted the greatest film character of all time, 
        beating James Bond and Scarlett O'Hara in the process.
        The most ratings received were movies from the 1990s and one from the 1970s, 
        where Shawshank Redmeption, The (1994) got the most ratings followed by Forrest Gump (1994).""")

        st.header("Top 10 Most Average Rated Movies")
        st.image("resources/imgs/Top 10 Most Average Rated Movies.png")
        st.write(""" We will be looking at the top 10 movies with the most average rating.
        Planet Earth II is a British nature documentary series produced by the BBC as a sequel to Planet Earth. 
        Cosmos is an American science documentary television series
        Planet Earth II (2016) has the highest average ratings followed by Planet Earth (2006)  """)
        
        st.header("Top 10 movies ratings ")
        st.image("resources/imgs/Top 10 movies ratings.png")
        st.write("""We will be looking at the total number of ratings per year.
        Note: This is for the total ratings ever made each year.
        The graph above depicts the ratings for this dataset by year (averaged year per user).
        It demonstrated that certain consumers have preferences for release years,
        which could be relevant for undertaking predictive modeling.We can observe that all ratings were made between 1995 and 2020, with no evident association between year and quantity of ratings created/collected within that time frame.
        The most ratings were received in the year 2016 with a total of 702 962.
        As we can see that the year 1998 received the least count of ratings with a total of 108 811.
         After 2014, the count of rating increased tremendously and started declining after 2016.""")

        st.header("Top 10 most Average Rated Movies")
        st.image("resources/imgs/Top 10 Most Average Rated Movies.png")
        st.write(""" We will be looking at the top 10 movies with the most average rating.
        Planet Earth II is a British nature documentary series produced by the BBC as a sequel to Planet Earth. 
        Cosmos is an American science documentary television series
        Planet Earth II (2016) has the highest average ratings followed by Planet Earth (2006)  """)

    if page_selection == "Model Performance":
        st.title("Empty for now")
        st.markdown("Some info here soon")


    if page_selection == "Solution Overview":
        st.title("Solution Overview")
        st.write("""We trained our data on several models, of which KNNwithmeans and SVD gave us the best RMSE. With SVD giving us 
the best performance of 0.81706. After going through the 8 datasets, we ended up scraping Genome_tag_scores, 
Genome_tags as well as we felt these did not add much value, aslo because in the ended we had not used them. \n
Recommenders are big business. From Amazon's online MegaStore, as it is estimated that about 25% of the purchases 
made on Amazon is due to ML item recommenders. Also Neflix and streaming gaint Youtube, even Google and other 
search engines. Getting new users and keeping them on your platform has become the new gold rush. \n """)

        st.image('resources/imgs/Global_Recommendation.jpeg', caption = """Global 
        Recommendation Engine 2020-2024(src : https://mms.businesswire.com/)""",use_column_width=True)
        st.write("""Recommendation engines are easy to build, to put together so much so that almost any business,
        can add them to their arsenal of tools to capture market share. Even Small to Medium Enterprises(SMEs) 
        as well as governments and even animal breeders can take advantage of systems like these. Be it deciding 
        which products to expand the business to, or deciding on prosopective employees for easier team cohesioin 
        as well as recommending dog breeds/types to clients according to the type of pet, behaviors and activity 
        levels of the pet the client would like. 
        \nThe early 1900s, with the rise of Ford motors' warehouse manufacturing, 
        the age of mass production was born. Then as the world was becoming richer and more and more peolple could 
        afford cars, the age of design and customisation began. Now we living through the contructions of the a new 
        age. The age of personalisation, where every client gets to experience tailor made services.

         \nAs we have indicated, there is an endless universe of what can be accomplished with a couple of lines of 
         code and a little bit of time collecting data. Because as we all know, knowledge is **power**.""")

    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.


if __name__ == '__main__':
    main()



