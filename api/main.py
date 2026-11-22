import os
from io import BytesIO
from functools import lru_cache

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from core.ingest import get_store, add_pdf
from core.graph import build_graph
from core.utils import cite

load_dotenv(override=True)

app = FastAPI(
    title="Lilies API",
    description="Bilingual PDF Question Answering API",
    version="1.0.0"
)


# --------------------------------------------------
# 1. Load ChromaDB once
# --------------------------------------------------

@lru_cache
def get_store_instance():
    return get_store(path="chroma_db")


# --------------------------------------------------
# 2. Build a graph for each selected PDF
# --------------------------------------------------

@lru_cache
def get_graph(source: str):
    store = get_store_instance()
    return build_graph(store, source=source)


# --------------------------------------------------
# 3. Request model
# --------------------------------------------------

class AskRequest(BaseModel):
    question: str
    source: str


# --------------------------------------------------
# 4. Health check
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "project": "Lilies"
    }


# --------------------------------------------------
# 5. List uploaded PDFs
# --------------------------------------------------

@app.get("/documents")
def get_documents():

    store = get_store_instance()

    data = store.get(include=["metadatas"])

    filenames = sorted({
        metadata["source"]
        for metadata in data["metadatas"]
        if metadata and metadata.get("source")
    })

    return {
        "documents": filenames,
        "count": len(filenames)
    }


# --------------------------------------------------
# 6. Upload PDF
# --------------------------------------------------

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty."
        )

    try:
        store = get_store_instance()

        pdf_file = BytesIO(content)

        chunks = add_pdf(
            store,
            pdf_file,
            file.filename
        )

        return {
            "filename": file.filename,
            "chunks_added": chunks,
            "status": "success"
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# --------------------------------------------------
# 7. Ask a question about a selected PDF
# --------------------------------------------------

@app.post("/ask")
def ask_question(request: AskRequest):

    question = request.question.strip()
    source = request.source.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    if not source:
        raise HTTPException(
            status_code=400,
            detail="Please select a PDF document."
        )

    try:

        store = get_store_instance()

        # Check if the selected PDF exists
        documents = store.get(
            where={"source": source},
            include=["metadatas"]
        )

        if not documents["ids"]:
            raise HTTPException(
                status_code=404,
                detail="The selected document was not found."
            )

        # Build a graph filtered by the selected PDF
        graph = get_graph(source)

        result = graph.invoke({
            "question": question
        })

        sources = []

        if result.get("relevant"):

            for doc in result["docs"]:

                sources.append({
                    "filename": doc.metadata["source"],
                    "page": doc.metadata["page"],
                    "citation": cite(doc.metadata),
                    "content": doc.page_content[:300]
                })

        return {
            "question": question,
            "selected_document": source,
            "answer": result["answer"],
            "sources": sources
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )