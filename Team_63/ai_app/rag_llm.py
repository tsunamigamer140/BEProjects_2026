import os
import json
import voyageai
from pinecone import Pinecone
from openai import OpenAI

# ---------------------------
# CONFIG (FROM AZURE ENV VARS)
# ---------------------------

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_HOST = os.environ.get("PINECONE_INDEX_HOST")
PINECONE_NAMESPACE = os.environ.get("PINECONE_NAMESPACE", "roadmaps")

VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY")

HF_API_KEY = os.environ.get("HF_API_KEY")
HF_BASE_URL = os.environ.get("HF_BASE_URL")
HF_MODEL = os.environ.get("HF_MODEL", "ganeshaMD/roadmap-ai")

TOP_K = int(os.environ.get("TOP_K", 5))


# ---------------------------
# VALIDATION (OPTIONAL BUT SAFE)
# ---------------------------

if not all([
    PINECONE_API_KEY,
    PINECONE_INDEX_HOST,
    VOYAGE_API_KEY,
    HF_API_KEY,
    HF_BASE_URL
]):
    raise RuntimeError("❌ One or more required environment variables are missing")


# ---------------------------
# INIT CLIENTS
# ---------------------------

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=PINECONE_INDEX_HOST)

vo = voyageai.Client(api_key=VOYAGE_API_KEY)

hf_client = OpenAI(
    base_url=HF_BASE_URL,
    api_key=HF_API_KEY
)


# ---------------------------
# EMBEDDING FUNCTION
# ---------------------------

def embed_query(text: str):
    try:
        res = vo.embed(
            model="voyage-3",
            texts=[text]
        )
        return res.embeddings[0]
    except Exception as e:
        print("Embedding error:", e)
        return None


# ---------------------------
# PINECONE RETRIEVAL
# ---------------------------

def retrieve_context(query: str):
    vector = embed_query(query)
    if vector is None:
        return []

    try:
        results = index.query(
            vector=vector,
            top_k=TOP_K,
            namespace=PINECONE_NAMESPACE,
            include_metadata=True
        )

        contexts = []
        for match in results.matches:
            if match.metadata and "text" in match.metadata:
                contexts.append(match.metadata["text"])
            else:
                contexts.append(json.dumps(match.metadata))

        return contexts

    except Exception as e:
        print("Pinecone Query Error:", e)
        return []


# ---------------------------
# FORMAT CONTEXT
# ---------------------------

def format_context(context_blocks):
    if not context_blocks:
        return "No useful context retrieved."

    return "\n---\n".join(context_blocks)


# ---------------------------
# HF LLM GENERATION
# ---------------------------

def generate_answer(query: str, context_text: str):

    prompt = f"""
You are an expert roadmap generator specializing in creating highly detailed learning plans.

Your task:
Generate a COMPLETE, DAY-WISE roadmap in a flowchart-style sequence. The roadmap must be extremely detailed and cover:
- Every topic
- Every sub-topic
- Tools required
- Concepts to master
- What to practice
- Expected outcomes
- Mini tasks or exercises
- Progression logic

STRICT FORMAT RULES:
- NO asterisks (*)
- NO markdown
- NO bold/italics
- Plain text only

STRUCTURE:

Day 1 → Main Topic
  Explanation
  Subtopics
  Tools
  Concepts
  Mini tasks
  Outcome

Continue for all days.

RAG CONTEXT:
{context_text}

USER QUESTION:
{query}

Generate the full roadmap:
"""

    try:
        resp = hf_client.chat.completions.create(
            model=HF_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content

    except Exception as e:
        print("LLM Error:", e)
        return "Error generating LLM answer."


# ---------------------------
# MAIN RAG FUNCTION
# ---------------------------

def rag_answer(query: str):
    try:
        context_blocks = retrieve_context(query)
        context = format_context(context_blocks)
        answer = generate_answer(query, context)
        return answer

    except Exception as e:
        return f"[RAG SYSTEM ERROR] {e}"
