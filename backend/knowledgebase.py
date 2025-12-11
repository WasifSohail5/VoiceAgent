# app/knowledgebase.py
"""
Knowledge Base - Upload documents, extract text, create embeddings
"""

import os
import uuid
import tempfile
import math
from typing import List

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks
from sqlalchemy.orm import Session
from pydantic import BaseModel
from datetime import datetime

import pdfplumber
import docx
from openai import OpenAI
from supabase import create_client

from main import (
    get_db, get_current_user, User, Agent, KnowledgeBase, Embedding,
    SUPABASE_URL, SUPABASE_KEY, OPENAI_API_KEY
)

router = APIRouter(prefix="/knowledge", tags=["Knowledge Base"])

# Clients
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)


# ===========================================
# SCHEMAS
# ===========================================
class KnowledgeBaseResponse(BaseModel):
    id: str
    agent_id: str
    filename: str
    file_type: str
    status: str
    chunk_count: int
    created_at: datetime

    class Config:
        from_attributes = True


# ===========================================
# HELPER FUNCTIONS
# ===========================================
def extract_text(file_path: str, file_type: str) -> str:
    """Extract text from PDF, DOCX, TXT"""

    if file_type == "pdf":
        with pdfplumber.open(file_path) as pdf:
            return "\n".join([page.extract_text() or "" for page in pdf.pages])

    elif file_type in ["docx", "doc"]:
        doc = docx.Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])

    else:  # txt or others
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks"""

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # Break at sentence if possible
        if end < len(text):
            last_period = chunk.rfind('.')
            if last_period > chunk_size * 0.5:
                chunk = chunk[:last_period + 1]
                end = start + last_period + 1

        chunks.append(chunk.strip())
        start = end - overlap

    return chunks


def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate OpenAI embeddings"""

    embeddings = []
    for text in texts:
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        embeddings.append(response.data[0].embedding)
    return embeddings


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity"""

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


def process_document(db: Session, kb_id: str):
    """Background task:  Extract text, chunk, create embeddings"""

    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
    if not kb:
        return

    try:
        kb.status = "processing"
        db.commit()

        # Download from Supabase
        file_data = supabase.storage.from_("knowledge-files").download(kb.storage_path)

        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{kb.file_type}") as tmp:
            tmp.write(file_data)
            tmp_path = tmp.name

        # Extract text
        text = extract_text(tmp_path, kb.file_type)
        kb.extracted_text = text

        # Chunk text
        chunks = chunk_text(text)
        kb.chunk_count = len(chunks)

        # Generate and store embeddings
        if chunks:
            embeddings = generate_embeddings(chunks)

            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                emb = Embedding(
                    knowledge_base_id=kb.id,
                    chunk_text=chunk,
                    chunk_index=idx,
                    embedding_vector=embedding,
                    metadata={"filename": kb.filename}
                )
                db.add(emb)

        kb.status = "completed"
        db.commit()

        # Cleanup
        os.remove(tmp_path)

    except Exception as e:
        kb.status = "failed"
        db.commit()
        print(f"Error processing document: {e}")


# ===========================================
# API ROUTES
# ===========================================
@router.post("/upload/{agent_id}", response_model=KnowledgeBaseResponse)
async def upload_document(
        agent_id: str,
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """Upload a document (PDF, DOCX, TXT) to knowledge base"""

    # Verify agent
    agent = db.query(Agent).filter(Agent.id == agent_id, Agent.user_id == current_user.id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Validate file type
    allowed = ["pdf", "docx", "doc", "txt"]
    file_ext = file.filename.split(".")[-1].lower()

    if file_ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Allowed:  {allowed}")

    try:
        # Upload to Supabase
        content = await file.read()
        storage_path = f"{current_user.id}/{agent_id}/{uuid.uuid4()}.{file_ext}"
        supabase.storage.from_("knowledge-files").upload(storage_path, content)

        # Save to DB
        kb = KnowledgeBase(
            agent_id=agent_id,
            filename=file.filename,
            file_type=file_ext,
            storage_path=storage_path,
            status="pending"
        )
        db.add(kb)
        db.commit()
        db.refresh(kb)

        # Process in background
        background_tasks.add_task(process_document, db, kb.id)

        return kb

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{agent_id}", response_model=List[KnowledgeBaseResponse])
def get_documents(
        agent_id: str,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """Get all documents for an agent"""

    agent = db.query(Agent).filter(Agent.id == agent_id, Agent.user_id == current_user.id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    return db.query(KnowledgeBase).filter(KnowledgeBase.agent_id == agent_id).all()


@router.get("/{agent_id}/status/{kb_id}")
def get_status(
        agent_id: str,
        kb_id: str,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """Check document processing status"""

    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Document not found")

    return {"id": kb.id, "filename": kb.filename, "status": kb.status, "chunks": kb.chunk_count}


@router.post("/{agent_id}/search")
def search_documents(
        agent_id: str,
        query: str,
        top_k: int = 5,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """Semantic search in knowledge base"""

    # Get query embedding
    response = openai_client.embeddings.create(model="text-embedding-3-small", input=query)
    query_embedding = response.data[0].embedding

    # Get all embeddings for agent
    kb_ids = [kb.id for kb in db.query(KnowledgeBase).filter(
        KnowledgeBase.agent_id == agent_id,
        KnowledgeBase.status == "completed"
    ).all()]

    embeddings = db.query(Embedding).filter(Embedding.knowledge_base_id.in_(kb_ids)).all()

    # Calculate similarity
    results = []
    for emb in embeddings:
        similarity = cosine_similarity(query_embedding, emb.embedding_vector)
        results.append({"text": emb.chunk_text, "similarity": similarity})

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return {"query": query, "results": results[:top_k]}


@router.delete("/{agent_id}/{kb_id}")
def delete_document(
        agent_id: str,
        kb_id: str,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """Delete a document"""

    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Document not found")

    # Delete from Supabase
    try:
        supabase.storage.from_("knowledge-files").remove([kb.storage_path])
    except:
        pass

    # Delete embeddings
    db.query(Embedding).filter(Embedding.knowledge_base_id == kb_id).delete()

    db.delete(kb)
    db.commit()

    return {"message": "Deleted successfully"}