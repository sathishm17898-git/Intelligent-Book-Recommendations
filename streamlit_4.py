import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

setwd = "D:/Sathish/AIML/Intelligent book recommendations/"
# Load artifacts
with open("D:/Sathish/AIML/Intelligent book recommendations/kmean_overall.pkl", "rb") as f:
    kMeans = pickle.load(f)

with open("D:/Sathish/AIML/Intelligent book recommendations/tf_genre.pkl", "rb") as f:
    genre_vectorizer = pickle.load(f)


from sentence_transformers import SentenceTransformer
desc_model = SentenceTransformer('all-MiniLM-L6-v2')


genre_sim = np.load("D:/Sathish/AIML/Intelligent book recommendations/genre_sim.npy")
data = pd.read_csv("D:/Sathish/AIML/Intelligent book recommendations/books with clusters.csv")


genre_matrix = genre_vectorizer.transform(data['Genre'])
desc_matrix = np.load("D:/Sathish/AIML/Intelligent book recommendations/desc_embed.npy")  # saved earlier

def sim_book(sim_matrix,book_index,k=5):
    similar_indices=sim_matrix[book_index].argsort()[::-1][1:k+1]
    return similar_indices


def hybrid_recommendations(data, sim_matrix, book_index, k=5):
    # Step 1: Find cluster of target book
    cluster_id = data.iloc[book_index]['Cluster_overall']
    
    # Step 2: Get all books in same cluster (excluding itself)
    cluster_books = data[data['Cluster_overall'] == cluster_id].index.tolist()
    cluster_books = [idx for idx in cluster_books if idx != book_index]
    
    # Step 3: Rank by cosine similarity inside the cluster
    sims = sim_matrix[book_index, cluster_books]
    top_indices = [cluster_books[i] for i in sims.argsort()[::-1][:k]]
    
    return top_indices

st.title("Book Recommendation System")

# User inputs
user_genre = st.text_input("Enter Genre:")
user_desc = st.text_area("Enter Description:")
user_rating = st.slider("Minimum Rating", 0.0, 5.0, 3.5, 0.1)
user_reviews = st.number_input("Minimum Number of Reviews", min_value=0, step=10, value=50)
user_price = st.number_input("Maximum Price", min_value=0, step=10, value=500)

if st.button("Recommend"):
    # Vectorize inputs
    genre_vec = genre_vectorizer.transform([user_genre])
    desc_vec = desc_model.encode([user_desc])

    # Compute similarities
    genre_sim = cosine_similarity(genre_vec, genre_matrix)[0]
    desc_sim = cosine_similarity(desc_vec.reshape(1, -1), desc_matrix)[0]

    # Hybrid score (weighted)
    alpha, beta = 0.5, 0.5
    hybrid_score = alpha * genre_sim + beta * desc_sim

    # Rank books
    top_indices = hybrid_score.argsort()[::-1]

    # Apply filters
    filtered = []
    for idx in top_indices:
        row = data.iloc[idx]
        if (row['Rating'] >= user_rating and
            row['Number of Reviews'] >= user_reviews and
            row['Price'] <= user_price):
            filtered.append(idx)
        if len(filtered) >= 5:  # top 5 after filtering
            break

    # Display results
    st.write("Recommended Books:")
    results=[]
    for idx in filtered:
        row = data.iloc[idx]
        results.append({
            "Title": row["Book Name"],
            "Author":row['Author'],
            "Genre": row["Genre"],
            "Rating": row["Rating"],
            "Reviews": row["Number of Reviews"],
            "Price": row["Price"]
        })
    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Display as table
    st.subheader("Recommended Books")
    st.dataframe(results_df)   # static table



