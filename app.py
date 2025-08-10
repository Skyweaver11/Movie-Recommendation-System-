import streamlit as st
import pandas as pd
from movie_recommendation_system import load_and_preprocess_data, get_recommendations

# Streamlit app UI
st.title('Movie Recommendation System')

st.write('Find similar movies by entering a movie name or uploading a CSV with movie titles.')

# Load and preprocess data (cached for performance)
@st.cache_data
def get_movie_data():
    return load_and_preprocess_data()

movies_data, similarity, vectorizer = get_movie_data()

# Option selection
option = st.selectbox('Choose Input Method', ('Enter Movie Name', 'Upload CSV'))

if option == 'Enter Movie Name':
    st.write('Enter the name of a movie to get recommendations.')
    
    # Movie name input
    movie_name = st.text_input('Movie Name', placeholder='e.g., The Dark Knight')
    
    # Recommend button
    if st.button('Get Recommendations'):
        if movie_name:
            close_match, recommendations = get_recommendations(movie_name, movies_data, similarity)
            if close_match:
                st.subheader(f'Recommendations for "{close_match}"')
                if recommendations:
                    # Display as a table
                    result_df = pd.DataFrame(recommendations, columns=['Rank', 'Movie Title'])
                    st.table(result_df)
                else:
                    st.warning('No recommendations found.')
            else:
                st.error(f'No close match found for "{movie_name}". Please check the spelling or try another movie.')
        else:
            st.error('Please enter a movie name.')

elif option == 'Upload CSV':
    st.write('Upload a CSV file with a single column "title" containing movie names.')
    
    # File uploader
    uploaded_file = st.file_uploader('Upload Movie Titles CSV', type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read the uploaded CSV
            input_data = pd.read_csv(uploaded_file)
            
            # Validate CSV
            if 'title' not in input_data.columns:
                st.error('CSV must contain a "title" column.')
            else:
                # Get recommendations for each movie
                results = []
                for movie_name in input_data['title']:
                    close_match, recommendations = get_recommendations(movie_name, movies_data, similarity)
                    if close_match:
                        for rank, title in recommendations:
                            results.append({
                                'Input Movie': movie_name,
                                'Closest Match': close_match,
                                'Recommended Movie': title,
                                'Rank': rank
                            })
                    else:
                        results.append({
                            'Input Movie': movie_name,
                            'Closest Match': 'No match found',
                            'Recommended Movie': 'None',
                            'Rank': '-'
                        })
                
                # Convert to DataFrame
                result_df = pd.DataFrame(results)
                
                # Display results
                st.subheader('Recommendation Results')
                st.write('The table below shows recommendations for each uploaded movie title.')
                st.dataframe(result_df)
                
                # Summary
                valid_recommendations = result_df[result_df['Closest Match'] != 'No match found']
                st.write(f'**Summary**: Recommendations found for {len(valid_recommendations["Input Movie"].unique())} out of {len(input_data)} input movies.')
                
                # Download results
                csv = result_df.to_csv(index=False)
                st.download_button(
                    label='Download Recommendations',
                    data=csv,
                    file_name='movies.csv',
                    mime='text/csv'
                )
        except Exception as e:
            st.error(f'Error processing file: {str(e)}')

st.write('**Note**: The system uses TF-IDF and cosine similarity based on genres, keywords, tagline, cast, and director. Ensure movie names match the dataset (e.g., check spelling).')