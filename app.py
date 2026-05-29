import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_data
def load_data():
    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")
    movies = movies.merge(credits, on='title')
    movies = movies[['movie_id','title','overview',
                     'genres','keywords','cast','crew']]
    
    def convert(obj):
        return [i['name'] for i in ast.literal_eval(obj)]
    
    def convert3(obj):
        return [i['name'] for i in ast.literal_eval(obj)][:3]
    
    def fetch_director(obj):
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                return [i['name']]
        return []
    
    movies['overview'] = movies['overview'].fillna('')
    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(convert3)
    movies['crew'] = movies['crew'].apply(fetch_director)
    movies['overview'] = movies['overview'].apply(lambda x: x.split())
    
    for col in ['genres','keywords','cast','crew']:
        movies[col] = movies[col].apply(
            lambda x: [i.replace(" ","") for i in x])
    
    movies['tags'] = (movies['overview'] + movies['genres'] + 
                     movies['keywords'] + movies['cast'] + 
                     movies['crew'])
    
    new_df = movies[['movie_id','title','tags']].copy()
    new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
    
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(new_df['tags']).toarray()
    similarity = cosine_similarity(vectors)
    
    return new_df, similarity

movies, similarity = load_data()

def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),
                  reverse=True, key=lambda x: x[1])[1:6]
    return [movies.iloc[i[0]].title for i in movies_list]

st.title('🎬 Movie Recommender System')
selected_movie = st.selectbox('Select a movie', movies['title'].values)

if st.button('Recommend'):
    recommendations = recommend(selected_movie)
    st.subheader('Top 5 Recommendations:')
    for movie in recommendations:
        st.write('✅', movie)
