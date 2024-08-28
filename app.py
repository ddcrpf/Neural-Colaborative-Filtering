import streamlit as st
import tensorflow as tf
import numpy as np

# Load model
model = tf.keras.models.load_model('ncf_model.h5')

# Title
st.title('Movie Recommendation System')

# User Input
user_id = st.number_input('Enter User ID:', min_value=1)
movie_id = st.number_input('Enter Movie ID:', min_value=1)

# Predict Button
if st.button('Predict'):
    prediction = model.predict([np.array([user_id]), np.array([movie_id])])
    st.write(f'Predicted Rating: {prediction[0][0]}')

# Recommend Movies Button
if st.button('Recommend Movies'):
    # This should include logic to get top N recommendations
    # For simplicity, I'm using a placeholder for recommended movies
    top_movies = recommend_top_movies(user_id)
    st.write('Top Movie Recommendations:')
    for movie in top_movies:
        st.write(movie)

def recommend_top_movies(user_id, top_n=10):
    # Placeholder logic for movie recommendations
    # You should replace this with actual logic to get top N movie recommendations
    # For example, you might need to use your model to predict ratings for all movies and then sort them.
    all_movies = range(1, 101)  # Assuming movie IDs range from 1 to 100
    ratings = []
    
    for movie in all_movies:
        pred = model.predict([np.array([user_id]), np.array([movie])])
        ratings.append((movie, pred[0][0]))
    
    # Sort movies by predicted rating in descending order
    sorted_movies = sorted(ratings, key=lambda x: x[1], reverse=True)
    
    # Get top N movies
    top_movies = [movie for movie, _ in sorted_movies[:top_n]]
    
    return top_movies
