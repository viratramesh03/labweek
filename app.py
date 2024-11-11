import os
import re
import base64
import requests
import git
import faiss
import numpy as np
import uvicorn
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException, Query, Form
from pydantic import BaseModel
from git.exc import GitCommandError
from typing import Dict, List
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
# Initialize the SentenceTransformer model for semantic search
model = SentenceTransformer('all-MiniLM-L6-v2')
# GitHub Personal Access Token for private repo access
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_API_RATE_LIMIT = 5000
requests_made = 0

# Confluence credentials and base URL
CONFLUENCE_USERNAME = os.getenv("CONFLUENCE_USERNAME")
CONFLUENCE_API_TOKEN = os.getenv("CONFLUENCE_API_TOKEN")
CONFLUENCE_BASE_URL = os.getenv("CONFLUENCE_BASE_URL")
CONFLUENCE_BASE_URL_PAGE = os.getenv("CONFLUENCE_BASE_URL_PAGE")
CONFLUENCE_USERCODE = os.getenv("CONFLUENCE_USERCODE")
# Function to check GitHub API rate limit
def check_rate_limit():
    global requests_made
    if requests_made >= GITHUB_API_RATE_LIMIT:
        raise HTTPException(status_code=429, detail="GitHub API rate limit reached. Try again later.")
    requests_made += 1

# Function to clone or pull private GitHub repositories
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

# Function to fetch markdown files from repositories
def fetch_from_repos(repos: Dict[str, str]):
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

# Define function to extract keywords, filtering out common filler words
def extract_keywords(sentence):
    pattern = r'\b(?:what|is|why|the|a|an|of|and|for|to|in|on|at|by|with|as|from)\b|[^\w\s]'
    keywords = re.sub(pattern, '', sentence, flags=re.IGNORECASE).split()
    print(keywords)
    return [word for word in keywords if word]

# Function to fetch documents from Confluence
def fetch_confluence_docs(query):
    # Tokenize and filter the query to extract keywords
    keywords = extract_keywords(query)
    if keywords:
        keyword_query = " OR ".join([f'text ~ "{word}"' for word in keywords])
    else:
        return {"message": "No valid keywords found in query."}

    auth_string = f"{CONFLUENCE_USERNAME}:{CONFLUENCE_API_TOKEN}"
    encoded_auth_string = base64.b64encode(auth_string.encode('utf-8')).decode('utf-8')
    
    headers = {
        "Authorization": f"Basic {encoded_auth_string}",
        "Content-Type": "application/json"
    }
    
    params = {
        "cql": keyword_query,
        "limit": 10
    }

    try:
        response = requests.get(f"{CONFLUENCE_BASE_URL}/content/search", headers=headers, params=params)
        response.raise_for_status()
        data = response.json()

        results = []
        for result in data.get("results", []):
            page_id = result["id"]
            title = result["title"]
            url = f"{CONFLUENCE_BASE_URL_PAGE}/spaces/~{CONFLUENCE_USERCODE}/pages/{page_id}"
            excerpt = result.get("excerpt", "")

            # If excerpt is empty, fetch the page content for a snippet
            if not excerpt:
                page_content_response = requests.get(
                    f"{CONFLUENCE_BASE_URL}/content/{page_id}?expand=body.view", headers=headers
                )
                if page_content_response.status_code == 200:
                    page_content_html = page_content_response.json().get("body", {}).get("view", {}).get("value", "")
                    # Use BeautifulSoup to remove HTML tags
                    if page_content_html:
                        soup = BeautifulSoup(page_content_html, "html.parser")
                        page_content_text = soup.get_text()
                        # Extract a snippet from the start of the content
                        excerpt = page_content_text[:200].strip() + "..."  # Adjust the length as needed

            results.append({
                "title": title,
                "url": url,
                "snippet": excerpt
            })

        print("Confluence Results:", results)
        return results
    except requests.exceptions.RequestException as e:
        print(f"Error fetching from Confluence: {e}")
        return []

# Function to vectorize documents
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

# Store vectors in FAISS for similarity search
def store_in_faiss(vectors):
    dimension = 384
    index = faiss.IndexFlatL2(dimension)
    vector_data = np.array([v[2] for v in vectors], dtype='float32')
    index.add(vector_data)
    doc_mapping = {i: (v[0], v[1], v[3]) for i, v in enumerate(vectors)}
    return index, doc_mapping

# Define repositories
repos = {
    "https://github.com/ciec-infra/labweek.git": "test-vector-labweek",
    "https://github.com/ciec-infra/labweek-test.git": "test-vector-labweek-test"
}

# Fetch and vectorize documents
docs = fetch_from_repos(repos)
vectors = vectorize_docs(docs)
index, doc_mapping = store_in_faiss(vectors)

# Set up FastAPI
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Vector API!"}

# Define the request body for search
class QueryRequest(BaseModel):
    query: str
    page: int = Query(1, gt=0)
    size: int = Query(3, gt=0)

# Cache for search results
cache = defaultdict(dict)

# Combined search function
@app.post("/search/")
async def search_docs(query: QueryRequest):
    if query.query in cache and query.page in cache[query.query]:
        return cache[query.query][query.page]

    try:
        # GitHub document search
        query_vector = model.encode(query.query).astype('float32')
        D, I = index.search(np.array([query_vector]), k=query.size * query.page)
        github_results = []

        for idx, i in enumerate(I[0]):
            if idx < query.size * query.page:
                if i in doc_mapping:
                    relative_path, content, repo_url = doc_mapping[i]
                    clean_repo_url = repo_url.replace(".git", "")
                    github_url = f"{clean_repo_url}/blob/main/{relative_path}"
                    snippet_start = max(content.lower().find(query.query.lower()) - 30, 0)
                    snippet_end = min(snippet_start + 60, len(content))
                    snippet = content[snippet_start:snippet_end].strip() + "..." if len(content) > 60 else content
                    github_results.append({"file_path": github_url, "snippet": snippet})

        # Confluence document search
        confluence_results = fetch_confluence_docs(query.query)

        # Combine results
        combined_results = github_results + confluence_results
        cache[query.query][query.page] = {"results": combined_results}

        if len(combined_results) == 0:
            return {"results": [], "message": "There is no document related to the search"}

        return {"results": combined_results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during search: {str(e)}")

@app.post("/slack_command")
async def handle_slack_command(
    token: str = Form(...),
    team_id: str = Form(...),
    team_domain: str = Form(...),
    channel_id: str = Form(...),
    channel_name: str = Form(...),
    user_id: str = Form(...),
    user_name: str = Form(...),
    command: str = Form(...),
    text: str = Form(...),
    response_url: str = Form(...)
):
    try:
        query_request = {"query": text, "page": 1, "size": 3}
        response = await search_docs(QueryRequest(**query_request))

        if response["results"]:
            # Separate GitHub and Confluence results
            github_results = [result for result in response["results"] if "github.com" in result.get('file_path', '')]
            confluence_results = [result for result in response["results"] if "atlassian.net" in result.get('url', '')]

            # Build response text for GitHub docs
            github_text = "*Github Docs:*\n" if github_results else ""
            for idx, result in enumerate(github_results, start=1):
                file_path = result.get('file_path', result.get('url'))
                file_name = file_path.split('/')[-1]
                snippet = result['snippet']

                # Make file name a hyperlink
                github_text += f"{idx}. *<{file_path}|{file_name}>*\n"
                github_text += f">```\n{snippet}\n```\n\n"

            # Build response text for Confluence docs
            confluence_text = "*Confluence Docs:*\n" if confluence_results else ""
            for idx, result in enumerate(confluence_results, start=1):
                url = result.get('url')
                title = result.get('title')
                snippet = result['snippet']

                # Make Confluence title a hyperlink
                confluence_text += f"{idx}. *<{url}|{title}>*\n"
                confluence_text += f">```\n{snippet}\n```\n\n"

            # Combine the GitHub and Confluence results into the response text
            response_text = github_text + confluence_text if github_results or confluence_results else "No documents found for your query."

        else:
            response_text = "There is no document related to the search."

        return {
            "response_type": "in_channel",
            "text": response_text
        }

    except Exception as e:
        return {"text": f"Error during search: {str(e)}"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8003)
