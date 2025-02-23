import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

# --- 1. Data Preparation ---
# Suppose 'df' is a large DataFrame with columns: 'user', 'item', 'rating'
# For demonstration, we use our sample data; in practice, load your large dataset.
data = {
    'user': ['Alice', 'Alice', 'Bob', 'Bob', 'Charlie', 'Charlie'],
    'item': ['Movie1', 'Movie2', 'Movie2', 'Movie3', 'Movie1', 'Movie3'],
    'rating': [5, 3, 4, 2, 2, 5]
}
df = pd.DataFrame(data)

# Create the user-item matrix and use a sparse representation.
ratings_matrix = df.pivot_table(index='user', columns='item', values='rating').fillna(0)
# Convert dense matrix to sparse CSR format
ratings_sparse = csr_matrix(ratings_matrix.values)
print("User-Item Matrix (dense):")
print(ratings_matrix)

# --- 2. Similarity Calculation on Sparse Data ---
# Note: cosine_similarity from sklearn can handle sparse inputs.
user_sim_sparse = cosine_similarity(ratings_sparse)
user_sim_df = pd.DataFrame(user_sim_sparse, index=ratings_matrix.index, columns=ratings_matrix.index)
print("\nUser Similarity Matrix (Cosine, sparse input):")
print(user_sim_df)

# --- 3. Rating Prediction Function ---
def predict_rating(target_user, target_item, ratings, similarity_df):
    """
    Predict the rating for target_item by target_user using a weighted average
    of ratings from users similar to target_user.
    """
    if ratings.loc[target_user, target_item] != 0:
        return ratings.loc[target_user, target_item]
    
    # Get indices of users who rated the target item
    rated_by = ratings[ratings[target_item] != 0].index
    sim_scores = similarity_df.loc[target_user, rated_by]
    item_ratings = ratings.loc[rated_by, target_item]
    
    if sim_scores.sum() == 0:
        return 0  # fallback if no similar users exist
    
    return (sim_scores * item_ratings).sum() / sim_scores.sum()

# --- 4. Generating Recommendations ---
def recommend_items(user, ratings, similarity_df, top_n=2):
    user_ratings = ratings.loc[user]
    missing_items = user_ratings[user_ratings == 0].index
    predictions = {item: predict_rating(user, item, ratings, similarity_df) for item in missing_items}
    return sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:top_n]

# Example: predict for 'Alice'
predicted = predict_rating('Alice', 'Movie3', ratings_matrix, user_sim_df)
print(f"\nPredicted rating for Alice on Movie3: {predicted:.2f}")

recs = recommend_items('Alice', ratings_matrix, user_sim_df, top_n=2)
print("\nTop recommendations for Alice:")
for item, score in recs:
    print(f"{item}: predicted rating = {score:.2f}")
