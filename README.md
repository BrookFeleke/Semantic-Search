# E-commerce Product Recommendation Semantic Search Tool

## Description
This project is a domain-specific semantic search tool tailored for e-commerce product recommendations. Users can search for products using natural language, and the tool provides relevant recommendations.

## Features
- Semantic search using pre-trained Sentence Transformers.
- Mock e-commerce product dataset.
- Filter and rank results based on similarity scores.
- Simple and interactive UI built with Streamlit.

## Requirements
- Python 3.7 or higher
- Required Python packages: `numpy`, `pandas`, `sentence-transformers`, `scikit-learn`, `streamlit`.

## Setup Instructions
1. Clone the repository and navigate to the project directory.
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Dataset
The dataset (`data/products.csv`) contains mock data with attributes:
- `id`
- `title`
- `description`
- `category`
- `tags`

## Example Queries
- "Find a mouse for gaming."
- "Eco-friendly yoga products."
- "Compact protective laptop sleeves."

## Output
The app ranks products based on semantic similarity to the user's query.

