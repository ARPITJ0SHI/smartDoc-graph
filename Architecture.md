

---

Final Combined Architecture (Clean + Interview-Grade)

Key Decisions (Resolved)

Orchestration: LangGraph (state machine, conditional routing)

Retrieval: HyDE + CRAG + Cross-Encoder Re-ranking

Vector DB: FAISS (Dockerized, persisted)

Memory: Full chat logs + summary memory

Fallbacks: Explicit graph node (not prompt-only)

IDs: ✅ Both task_id + document_id (correct separation)

Observability: Minimal (no Flower)

Chunking: ✅ 1000 / 200 (better semantic continuity)

Code Structure: Modular (services + workflows)

LLM Usage: Single provider (OpenAI), model configurable



---

1. High-Level Architecture

Two Pipelines

A. Async Ingestion Pipeline

Upload → Queue → Worker → Chunk → Embed → Store


B. Sync Query (RAG) Pipeline

Query → LangGraph → Retrieval → Filtering → Answer



---

System Components

FastAPI → API layer

Redis → Queue + broker

Celery → Async workers

PostgreSQL → metadata + chat + memory

FAISS (Docker volume) → vector store

LangGraph → orchestration engine

OpenAI → all LLM tasks

SentenceTransformers → embeddings



---

2. Final Workflow

Document Upload (Async)

1. User uploads file


2. FastAPI:

saves file

creates document_id

creates task_id

status = PENDING



3. Push task → Redis


4. Worker:

extract text

chunk (1000 / 200)

embed

store in FAISS



5. Update DB:

status = COMPLETED





---

Query Pipeline (LangGraph)

Graph Flow

User Query
   ↓
[Load Memory]
   ↓
[HyDE Generation]
   ↓
[Retrieve FAISS]
   ↓
[CRAG Evaluation]
   ↓ (conditional)
   ├── valid → [Re-rank]
   │              ↓
   │        [Generate Answer]
   │              ↓
   │         [Update Memory]
   │              ↓
   │            RETURN
   │
   └── invalid → [Fallback Node]
                      ↓
                   RETURN


---

3. LangGraph State (Final)

class RAGState(TypedDict):
    query: str
    session_id: str

    chat_summary: str
    chat_history: List[Dict]

    hyde_doc: str
    retrieved_chunks: List[Document]
    reranked_chunks: List[Document]

    final_answer: str
    fallback: bool


---

4. Retrieval Pipeline (Final)

Step 1: HyDE

LLM generates hypothetical answer

Improves embedding quality


Step 2: Retrieval (FAISS)

Top-K = 10–20 chunks


Step 3: CRAG Filtering

LLM classifies:

Relevant

Irrelevant

Ambiguous



→ If no relevant → fallback node

Step 4: Re-ranking (Cross Encoder)

Model:

cross-encoder/ms-marco-MiniLM-L-6-v2


→ Keep top 3–5 chunks

Step 5: Final Generation

Prompt includes:

reranked chunks

summary memory

strict grounding instruction



---

5. Database Schema (Final)

Documents

Document:
- id (uuid)
- filename
- file_path
- status (PENDING | PROCESSING | COMPLETED | FAILED)
- error_message
- created_at


---

Tasks (NEW — important fix)

Task:
- id (uuid)
- document_id (fk)
- status
- created_at
- updated_at


---

Chat

ChatSession:
- id
- summary_memory
- created_at

ChatMessage:
- id
- session_id
- role
- content
- created_at


---

6. API Design (Final Clean Version)

Upload

POST /api/v1/documents/upload

Response:
{
  "document_id": "uuid",
  "task_id": "uuid",
  "status": "PENDING"
}


---

Task Status

GET /api/v1/tasks/{task_id}

{
  "status": "COMPLETED",
  "document_id": "uuid"
}


---

Chat

POST /api/v1/chat

{
  "session_id": "uuid",
  "query": "..."
}

Response:

{
  "answer": "...",
  "sources": [
    {
      "document_id": "...",
      "page": 3,
      "chunk_id": "..."
    }
  ]
}


---

7. Storage Layout

data/
├── uploads/
├── faiss/
│   ├── index.faiss
│   └── metadata.pkl


---

8. Critical Improvements (Added Beyond Both Designs)

✅ 1. ID Separation (Important Fix)

task_id → async tracking

document_id → actual entity


Most candidates miss this.


---

✅ 2. Deterministic Fallback Node

Instead of relying on prompts:

system explicitly branches

eliminates hallucination paths



---

✅ 3. Dual Memory Strategy

Raw messages → stored

Summary → used in prompt


→ best balance of:

observability

token efficiency



---

✅ 4. Controlled Context Size

Final prompt uses:

top 3–5 chunks only

summary instead of full history


→ prevents latency + cost explosion


---

✅ 5. FAISS Persistence Strategy

Load once at worker startup

Save after batch updates

Avoid reloading per request



---

9. Known Tradeoffs (Explicit)

FAISS Limitation

Not horizontally scalable

No filtering by metadata at scale


→ acceptable for:

assignment

single-node systems



---

Re-ranking Cost

Adds ~50–150ms latency

Improves accuracy significantly



---

LangGraph Complexity

Harder to implement

Much easier to debug + extend later



---

10. Final Architecture Summary (What You Built)

This is a:

> Production-grade RAG system with

async ingestion

graph-based retrieval orchestration

hallucination control (CRAG + fallback)

semantic retrieval optimization (HyDE)

precision layer (re-ranking)

scalable memory handling





---
