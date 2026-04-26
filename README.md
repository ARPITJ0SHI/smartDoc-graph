# Smart Document Q&A System

A production-grade asynchronous RAG API that allows users to upload documents (PDF/DOCX) and ask natural language questions. It features structural Markdown extraction, LangGraph orchestration, conversational memory, and robust hallucination fallback mechanisms.

**Live Deployment:** [INSERT DEPLOYMENT LINK HERE]

---

## 🎯 Design Decisions

### 1. Ingestion & Chunking Strategy
Instead of extracting raw text sequentially and arbitrarily splitting it by character count, the system converts PDFs and DOCX files into **Markdown**. 
- It uses LangChain's `MarkdownHeaderTextSplitter` to chunk documents based on semantic boundaries (e.g., `# Header 1`, `## Header 2`).
- **Chunk Size Defaults:** We use a base size of **2000 characters (approx. 500 tokens)** with a **400 character (20%) overlap**. Research shows 500 tokens is the optimal "Goldilocks zone" for RAG: large enough to capture complete thoughts (analytical queries), but small enough to maintain precise retrieval (factoid queries).
- This preserves tables, ensures topics are not split mid-thought, and injects the header hierarchy into the FAISS metadata so the LLM knows exactly which section of a document a chunk belongs to.

### 2. Retrieval & Re-ranking
Standard vector search (FAISS) is fast but relies purely on dense semantic proximity, which can miss nuanced context.
- **HyDE (Hypothetical Document Embeddings):** Before querying FAISS, the LLM generates a hypothetical answer to the user's query, which significantly improves the mathematical quality of the embedding search.
- **Cross-Encoder Re-ranking:** We retrieve 10-20 chunks from FAISS, but then pass them through a Cross-Encoder (e.g., `ms-marco-MiniLM-L-6-v2`). The Cross-Encoder deeply analyzes the exact word-level relationship between the query and the chunk, stripping the 20 results down to the **absolute top 3 most relevant chunks** to feed the LLM.

### 3. Orchestration (LangGraph)
The retrieval pipeline is not a simple linear chain; it is a state machine built with **LangGraph**.
- This enables complex, conditional logic.
- It seamlessly manages thread-level conversational memory using `MemorySaver`.

### 4. Hallucination Control (CRAG - Corrective RAG)
Retrieval engines often pull irrelevant chunks. Before generating an answer, the retrieved chunks are passed through a **CRAG Evaluation Node**.
- The LLM acts as a strict judge, grading each chunk as `RELEVANT`, `IRRELEVANT`, or `AMBIGUOUS`.
- If no chunks pass the threshold, LangGraph explicitly routes to a deterministic **Fallback Node** (e.g., "I don't have enough information"), completely neutralizing hallucinations.

### 5. Asynchronous Processing
Document ingestion is computationally heavy (extracting text, chunking, hitting embedding APIs, updating FAISS).
- FastAPI immediately accepts the upload and returns a `task_id`.
- **Celery + Redis** pick up the job in the background, update the database to `PROCESSING`, and eventually to `COMPLETED`, ensuring the API never blocks.

### 6. LLM Provider
*Note: Due to our infrastructure requirements, we opted to use the highly performant **Google Gemini API** (`gemini-flash-lite-latest` and `gemini-embedding-2`) rather than OpenAI. The architecture is provider-agnostic and relies on LangChain standard bindings.*

---

## 🚀 Local Setup

### Prerequisites
- Docker and Docker Compose
- A Gemini API Key (or OpenAI key if you change the config)

### 1. Configure Environment
Rename `.env.example` to `.env` and insert your API key:
```bash
cp .env.example .env
```
Ensure your `GOOGLE_API_KEY` is set inside the `.env` file.

### 2. Start the Stack
Run everything with a single command. This will spin up the FastAPI server, PostgreSQL, Redis, and the Celery worker.
```bash
docker-compose up --build -d
```

The API will be available at `http://localhost:8000`. You can view the interactive Swagger docs at `http://localhost:8000/docs`.

---

## 🧪 Testing the API

We have included sample documents in the `/sample_docs` folder.

### 1. Upload a Document
Upload a document to the async processing queue:
```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -F "file=@./sample_docs/sample_medical_doc.pdf"
```
**Response:**
```json
{
  "document_id": "doc-uuid",
  "task_id": "task-uuid",
  "status": "PENDING"
}
```

### 2. Check Task Status
```bash
curl -X GET "http://localhost:8000/api/v1/tasks/<task-uuid>"
```

### 3. Ask a Question
Once the task is `COMPLETED`, ask a question. To test conversational memory, use the same `session_id` in subsequent requests!
```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "session-uuid-1234",
    "query": "What are the common side effects listed in the document?"
  }'
```
