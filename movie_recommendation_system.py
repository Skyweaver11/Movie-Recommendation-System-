# -*- coding: utf-8 -*-
"""Copy of Project 18. Movie Recommendation System using Machine Learning with Python.ipynb

Adapted from Colab notebook:
    https://colab.research.google.com/drive/15KoIZU5gzK3AbIyZV2-VmQ4AoWTSUZVc
"""

import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_and_preprocess_data(file_path='movies.csv'):
    """Load and preprocess the movie dataset."""
    # Load the dataset
    movies_data = pd.read_csv(file_path)
    
    # Check for required columns
    required_columns = ['index', 'title', 'genres', 'keywords', 'tagline', 'cast', 'director']
    if not all(col in movies_data.columns for col in required_columns):
        raise ValueError("Dataset must contain columns: index, title, genres, keywords, tagline, cast, director")
    
    # Select relevant features
    selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
    
    # Replace null values with empty string
    for feature in selected_features:
        movies_data[feature] = movies_data[feature].fillna('')
    
    # Combine features into a single string
    combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + \
                       movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director']
    
    # Convert text data to feature vectors
    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(combined_features)
    
    # Compute cosine similarity
    similarity = cosine_similarity(feature_vectors)
    
    return movies_data, similarity, vectorizer

def get_recommendations(movie_name, movies_data, similarity, top_n=10):
    """Get movie recommendations based on input movie name."""
    # Get list of all movie titles
    list_of_all_titles = movies_data['title'].tolist()
    
    # Find close match for the movie name
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles, n=1)
    if not find_close_match:
        return None, []
    
    close_match = find_close_match[0]
    
    # Find the index of the movie
    index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
    
    # Get similarity scores
    similarity_score = list(enumerate(similarity[index_of_the_movie]))
    
    # Sort movies by similarity score
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)
    
    # Get top N recommendations
    recommendations = []
    for i, movie in enumerate(sorted_similar_movies[1:top_n+1], 1):  # Skip the input movie
        index = movie[0]
        title = movies_data[movies_data.index == index]['title'].values[0]
        recommendations.append((i, title))
    
    return close_match, recommendations

if __name__ == "__main__":
    # Example usage
    movies_data, similarity, _ = load_and_preprocess_data()
    movie_name = "The Dark Knight"  # Example movie
    close_match, recommendations = get_recommendations(movie_name, movies_data, similarity)
    if close_match:
        print(f"Closest match for '{movie_name}': {close_match}")
        print("Movies suggested for you:\n")
        for i, title in recommendations:
            print(f"{i}. {title}")
    else:
        print("No close match found.")
