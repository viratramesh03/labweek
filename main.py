import os
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Form
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
import uvicorn
from collections import defaultdict
from github_fetch import fetch_from_repos, vectorize_docs, store_in_faiss, search_in_faiss
from confluence_fetch import fetch_and_vectorize_confluence_docs, search_confluence

# Initialize the SentenceTransformer model for semantic search
model = SentenceTransformer('all-MiniLM-L6-v2')
# Set up FastAPI
app = FastAPI()

# Load repositories and vectorize documents
repos = {
    "https://github.com/ciec-infra/labweek.git": "test-vector-labweek",
    "https://github.com/ciec-infra/labweek-test.git": "test-vector-labweek-test"
}

# Fetch and vectorize documents from GitHub
docs = fetch_from_repos(repos)
vectors = vectorize_docs(docs)
index, doc_mapping = store_in_faiss(vectors)

# Initialize and vectorize Confluence documents
fetch_and_vectorize_confluence_docs()

# Cache for search results
cache = defaultdict(dict)

# Request body for search
class QueryRequest(BaseModel):
    query: str
    page: int = Query(1, gt=0)
    size: int = Query(3, gt=0)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Vector API!"}

@app.post("/search/")
async def search_docs(query: QueryRequest):
    if query.query in cache and query.page in cache[query.query]:
        return cache[query.query][query.page]

    try:
        # GitHub document search
        query_vector = model.encode(query.query).astype('float32')
        github_results = search_in_faiss(query_vector, index, doc_mapping, query.query, k=query.size * query.page, threshold=1.5)

        # Confluence document search
        confluence_results = search_confluence(query_vector, k=query.size * query.page, threshold=1.5)

        # Combine results
        combined_results = {
            "github_results": github_results,
            "confluence_results": confluence_results
        }

        cache[query.query][query.page] = combined_results
        return combined_results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# Handle Slack commands
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
        # Prepare query request for search
        query_request = {"query": text, "page": 1, "size": 3}
        response = await search_docs(QueryRequest(**query_request))

        if response["github_results"] or response["confluence_results"]:
            # Separate GitHub and Confluence results
            github_results = response["github_results"]
            confluence_results = response["confluence_results"]

            # Build response text for GitHub docs
            github_text = "*Github Docs:*\n" if github_results else ""
            for idx, result in enumerate(github_results, start=1):
                file_path = result.get('url', result.get('file_path'))
                file_name = file_path.split('/')[-1]
                snippet = result['snippet']
                score = result['score']  # The score is extracted from the search results

                # Make file name a hyperlink
                github_text += f"{idx}. *<{file_path}|{file_name}>* (Score: {score:.2f})\n"  # Append the score
                github_text += f">```\n{snippet}\n```\n\n"

            # Build response text for Confluence docs
            confluence_text = "*Confluence Docs:*\n" if confluence_results else ""
            for idx, result in enumerate(confluence_results, start=1):
                url = result.get('url')
                title = result.get('title')
                snippet = result['snippet']
                score = result['score']  # Adding score to the Confluence result

                # Make Confluence title a hyperlink
                confluence_text += f"{idx}. *<{url}|{title}>* (Score: {score:.2f})\n"  # Append the score
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
    uvicorn.run(app, host="0.0.0.0", port=8003)
