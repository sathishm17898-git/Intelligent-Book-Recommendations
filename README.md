This project builds a content-based book recommendation system using NLP techniques, clustering, and embeddings. 
It processes book descriptions and genres, learns semantic representations, and recommends similar books based on cluster and within each cluster cosine similarity.
Evaluation metrics like Precision@k are implemented to measure recommendation quality
**Description about the files:**
- Audible_Catlog.csv, Audible_Catlog_Advanced_Features.csv - raw datasets
- Project_4.ipynb - python script
- data.csv - cleaned csv
- streamlit_4.py - streamlit UI script
- **Data Cleaning & Preprocessing**
  Regex parsing of ranks and genres.
  Text normalization (lowercasing, punctuation removal, stopword filtering).
- **Embeddings**
  Word2Vec embeddings for descriptions and genres.
  TF‑IDF vectorization for sparse text representation.
  Sentence BERT model - all-MiniLM-L6-v2 (hugging face)
- **Clustering & Visualization**
  PCA for dimensionality reduction.
  Scatter plots with cluster coloring.
  Within each cluster books are grouped based on the cosine similarity on 'Genre'
- **Recommendation Engine**
  Cluster-based recommendations sorted by rating.
  Precision@k evaluation using genre overlap.
- **Evaluation**
  Precision metric implemented.
  Extendable to Recall@k and F1 score.
- **Deployment**
  Deployed in Streamlit ui to recommded the books based on the favourite genre of the user.
