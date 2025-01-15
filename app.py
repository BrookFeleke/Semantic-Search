import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load and preprocess dataset
@st.cache_data
def load_data():
    data = pd.read_csv("data/products.csv")
    return data

# Compute embeddings
@st.cache_resource
def compute_embeddings(data, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(data["description"] + " " + data["tags"])
    return model, embeddings

# Perform semantic search
def semantic_search(query, data, embeddings, model, top_k=5):
    query_embedding = model.encode([query])
    scores = cosine_similarity(query_embedding, embeddings)[0]
    data["score"] = scores
    results = data.sort_values(by="score", ascending=False).head(top_k)
    return results

# Streamlit UI
def main():
    st.title("E-commerce Product Recommendation Search")
    st.write("Search for products using natural language queries!")

    # Load dataset and model
    data = load_data()
    model, embeddings = compute_embeddings(data)

    # User input
    query = st.text_input("Enter your search query:", "")

    if query:
        results = semantic_search(query, data, embeddings, model)
        st.write("Top Recommendations:")
        for _, row in results.iterrows():
            st.write(f"**{row['title']}** ({row['category']}) - {row['description']}")

if __name__ == "__main__":
    main()
