import os
import base64
import requests
from fastapi import HTTPException
from bs4 import BeautifulSoup
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Confluence credentials and base URL
CONFLUENCE_USERNAME = os.getenv("CONFLUENCE_USERNAME")
CONFLUENCE_API_TOKEN = os.getenv("CONFLUENCE_API_TOKEN")
CONFLUENCE_BASE_URL = os.getenv("CONFLUENCE_BASE_URL")
CONFLUENCE_BASE_URL_PAGE = os.getenv("CONFLUENCE_BASE_URL_PAGE")
CONFLUENCE_USERCODE = os.getenv("CONFLUENCE_USERCODE")

# Initialize the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# FAISS index and document mapping for Confluence
confluence_index = None
confluence_doc_mapping = {}

# Function to fetch documents from Confluence and vectorize them
def fetch_and_vectorize_confluence_docs():
    global confluence_index, confluence_doc_mapping
    confluence_docs = []

    auth_string = f"{CONFLUENCE_USERNAME}:{CONFLUENCE_API_TOKEN}"
    encoded_auth_string = base64.b64encode(auth_string.encode('utf-8')).decode('utf-8')

    headers = {
        "Authorization": f"Basic {encoded_auth_string}",
        "Content-Type": "application/json"
    }

    params = {
        "cql": "type=page",  # Fetch all pages for vectorizing
        "limit": 100  # Adjust limit as needed
    }

    try:
        response = requests.get(f"{CONFLUENCE_BASE_URL}/content/search", headers=headers, params=params)
        response.raise_for_status()
        data = response.json()

        for result in data.get("results", []):
            page_id = result["id"]
            title = result["title"]
            url = f"{CONFLUENCE_BASE_URL_PAGE}/spaces/~{CONFLUENCE_USERCODE}/pages/{page_id}"
            snippet = result.get("excerpt", "")

            # If no snippet is provided, fetch the page content for a snippet
            if not snippet:
                page_content_response = requests.get(
                    f"{CONFLUENCE_BASE_URL}/content/{page_id}?expand=body.view", headers=headers
                )
                if page_content_response.status_code == 200:
                    page_content_html = page_content_response.json().get("body", {}).get("view", {}).get("value", "")
                    if page_content_html:
                        soup = BeautifulSoup(page_content_html, "html.parser")
                        snippet = soup.get_text()[:200].strip() + "..."

            confluence_docs.append({
                "title": title,
                "url": url,
                "snippet": snippet
            })

        # Vectorize documents and store in FAISS index
        vectors = [model.encode(doc['snippet']).astype('float32') for doc in confluence_docs]
        dimension = model.get_sentence_embedding_dimension()
        confluence_index = faiss.IndexFlatL2(dimension)
        confluence_index.add(np.array(vectors))

        confluence_doc_mapping = {i: doc for i, doc in enumerate(confluence_docs)}
        print("Confluence documents have been vectorized and stored in FAISS index.")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching from Confluence: {e}")
        return []

# Function to search Confluence documents using FAISS
def search_confluence(query_vector, k=3, threshold=1.5):
    if confluence_index is None:
        raise HTTPException(status_code=500, detail="Confluence index is not initialized.")

    D, I = confluence_index.search(np.array([query_vector]), k)
    results = []
    for idx, i in enumerate(I[0]):
        score = D[0][idx]
        if score < threshold:
            doc = confluence_doc_mapping[i]
            results.append({
                "title": doc["title"],
                "url": doc["url"],
                "snippet": doc["snippet"],
                "score": score
            })
    return results
