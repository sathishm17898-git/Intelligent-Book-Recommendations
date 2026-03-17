import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

setwd = "D:/Sathish/AIML/Intelligent book recommendations/"
# Load artifacts
with open("D:/Sathish/AIML/Intelligent book recommendations/data.pkl", "rb") as f:
    data = pickle.load(f)

with open("D:/Sathish/AIML/Intelligent book recommendations/cluster_genre_sim.pkl", "rb") as f:
    cluster_genre_sim = pickle.load(f)


def recommend_by_genre_rating(user_genres, user_rating, k=5):
    # Normalize dataset genres
    data['Genre'] = data['Genre'].str.lower().str.strip()

    # If user selected nothing, return None
    if not user_genres:
        return None

    # Take the first selected genre (string, not list)
    user_genre = user_genres[0].lower().strip()

    # Filter by genre using str.contains
    filtered = data[data['Genre'].str.contains(user_genre, case=False, na=False)]
    filtered = filtered[filtered['Rating'] >= user_rating]

    if filtered.empty:
        return None

    # Pick the first book as reference
    book_index = filtered.index[0]
    chosen_book = data.loc[[book_index],['Book Name','Author','Description','Number of Reviews','Rating','Genre','Cluster']]
    cluster_id = data.loc[book_index, 'Cluster']
    cluster_info = cluster_genre_sim[cluster_id]
    cluster_indices = list(cluster_info['indices'])
    sim_mat = cluster_info['sim_mat']

    pos = cluster_indices.index(book_index)
    scores = sim_mat[pos]

    ranked_positions = scores.argsort()[::-1]
    ranked_positions = [i for i in ranked_positions if cluster_indices[i] != book_index]

    top_indices = [cluster_indices[i] for i in ranked_positions[:k]]
    recs = data.loc[top_indices, ['Book Name','Author','Description','Number of Reviews','Rating','Genre']]
    return chosen_book, recs

# Build unique genre list for selectbox/multiselect
all_genres = (
    data['Genre']
    .str.split(',')
    .explode()
    .str.strip()
    .str.lower()
    .unique()
)
all_genres = sorted(all_genres)

st.title("Book Recommendation System")

pages = st.sidebar.radio("Go to",
    ['Home','Search book']
)
if pages == "Home":
    st.write("Intelligent Book Recommendations")
    st.write("""
            - System recommends the books based on the user genre and rating.
            - Search your favorite genre in the genre search box 
""")
else:
    user_genres = st.multiselect("Select a genre:", options=all_genres)
    user_rating = st.number_input("Enter minimum rating", min_value=0.0, max_value=5.0, value=4.5, step=0.1)
    k = st.number_input("Number of recommendations", min_value=1, max_value=20, value=5, step=1)

    if st.button("Recommend"):
        chosen_book,recs = recommend_by_genre_rating(user_genres, user_rating, k)
        if recs is None or recs.empty:
            st.warning("No books found for the selected genre and rating.")
        else:
            st.write("### Recommendations")
            st.dataframe(recs,use_container_width=True)
if st.checkbox("Show the chosen book"):
    chosen_book,recs = recommend_by_genre_rating(user_genres, user_rating, k)
    st.write("### Chosen Book (Reference)")
    st.dataframe(chosen_book[['Book Name','Author','Description','Number of Reviews','Rating','Genre','Cluster']])
