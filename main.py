import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from tree_rag import TreeRAG

# 1. Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is missing. Please check your .env file.")

# 2. Initialize the FastAPI app
app = FastAPI(
    title="Medical Tree RAG API",
    description="Backend API for the Vectorless Medical Tree RAG system",
    version="1.0"
)

# 3. Initialize the RAG system globally
# This ensures medical_tree.json and the BM25 index are only loaded once at startup,
# making subsequent API calls much faster.
rag = TreeRAG(
    tree_file_path="medical_tree.json",
    gemini_api_key=api_key,
    model="gemini-2.5-flash"
)

# 4. Define the exact structure of incoming requests
class ChatRequest(BaseModel):
    query: str

# 5. Create a health check endpoint (optional but helpful)
@app.get("/")
async def root():
    return {"message": "Medical RAG API is running. Send POST requests to /chat"}

# 6. Create the main chat endpoint
@app.post("/chat")
async def ask_question(request: ChatRequest):
    try:
        # Pass the incoming query to your existing .ask() method
        answer = rag.ask(request.query)
        
        # Return the answer (FastAPI automatically converts this to JSON)
        return {"answer": answer}
        
    except Exception as e:
        # If something breaks, return a standard 500 server error
        raise HTTPException(status_code=500, detail=str(e))
