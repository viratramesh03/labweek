import os
import git
import requests
from git.exc import GitCommandError
from fastapi import HTTPException
from sentence_transformers import SentenceTransformer
import base64
import re
import faiss
import numpy as np
from typing import Dict, List
from collections import defaultdict
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Initialize the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# GitHub API Rate Limit
GITHUB_API_RATE_LIMIT = 5000
requests_made = 0
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

def check_rate_limit():
    global requests_made
    if requests_made >= GITHUB_API_RATE_LIMIT:
        raise HTTPException(status_code=429, detail="GitHub API rate limit reached. Try again later.")
    requests_made += 1

def clone_or_pull_repo(repo_url, local_path):
    try:
        if repo_url.startswith("https://github.com/"):
            auth_repo_url = repo_url.replace("https://github.com/", f"https://{GITHUB_TOKEN}@github.com/")
        else:
            auth_repo_url = repo_url

        if not os.path.exists(local_path):
            print(f"Cloning repository {repo_url} to {local_path}")
            check_rate_limit()
            git.Repo.clone_from(auth_repo_url, local_path)
        else:
            print(f"Pulling latest changes from {repo_url}")
            repo = git.Repo(local_path)
            check_rate_limit()
            repo.remotes.origin.pull()

    except GitCommandError as e:
        raise HTTPException(status_code=500, detail=f"Error with Git command: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

def fetch_from_repos(repos: Dict[str, str]) -> List[Dict]:
    docs = []
    for repo_url, local_path in repos.items():
        try:
            clone_or_pull_repo(repo_url, local_path)
            for root, dirs, files in os.walk(local_path):
                for file in files:
                    if file.endswith(".md"):
                        file_path = os.path.join(root, file)
                        relative_path = os.path.relpath(file_path, start=local_path)
                        docs.append((relative_path, file_path, repo_url))
        except HTTPException as e:
            print(f"Skipping repository {repo_url}: {e.detail}")
    return docs

def vectorize_docs(docs):
    vectors = []
    for relative_path, file_path, repo_url in docs:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                vector = model.encode(content)
                vectors.append((relative_path, content, vector, repo_url))
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    return vectors

def store_in_faiss(vectors):
    dimension = 384
    index = faiss.IndexFlatL2(dimension)
    vector_data = np.array([v[2] for v in vectors], dtype='float32')
    index.add(vector_data)
    doc_mapping = {i: (v[0], v[1], v[3]) for i, v in enumerate(vectors)}
    return index, doc_mapping

def search_in_faiss(query_vector, index, doc_mapping, query_text, k=3, threshold=1.5):
    """
    Perform a FAISS search and filter results based on the threshold score.
    """
    D, I = index.search(np.array([query_vector]), k)
    results = []

    for idx, i in enumerate(I[0]):
        score = D[0][idx]  # Distance is used as score (lower is better in L2)
        if score < threshold:
            relative_path, content, repo_url = doc_mapping[i]
            clean_repo_url = repo_url.replace(".git", "")
            github_url = f"{clean_repo_url}/blob/main/{relative_path}"
            snippet_start = max(content.lower().find(query_text.lower()) - 50, 0)
            snippet_end = snippet_start + 200
            results.append({
                "title": f"{relative_path}",
                "url": github_url,
                "snippet": content[snippet_start:snippet_end],
                "score": score
            })
    
    return results
