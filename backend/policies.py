# app/policies.py
"""
Policies/Rules - Add rules, AI analysis, file upload
"""

import os
import uuid
import json
import tempfile
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from pydantic import BaseModel
from datetime import datetime

from openai import OpenAI
from supabase import create_client

from main import (
    get_db, get_current_user, User, Agent, Policy,
    SUPABASE_URL, SUPABASE_KEY, OPENAI_API_KEY
)
from knowledgebase import extract_text

router = APIRouter(prefix="/policies", tags=["Policies & Rules"])

# Clients
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)


# ===========================================
# SCHEMAS
# ===========================================
class PolicyCreate(BaseModel):
    title: str
    content: str
    category: Optional[str] = None
    priority: Optional[int] = 1


class PolicyResponse(BaseModel):
    id: str
    agent_id: str
    title: str
    content: str
    category: Optional[str]
    priority: int
    is_active: bool
    analyzed_summary: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


# ===========================================
# HELPER FUNCTIONS
# ===========================================
def analyze_policy(content: str) -> dict:
    """AI analysis of policy - summary and category suggestion"""

    prompt = f"""Analyze this policy/rule for a voice AI agent. 
Provide: 
1. Brief summary (2-3 sentences)
2. Category (greeting, closing, escalation, faq, compliance, general)
3. Key points

Policy: 
{content}

Respond in JSON: 
{{"summary":  "...", "category": "...", "key_points": [".. .", "... "]}}
"""

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)


# ===========================================
# API ROUTES
# ===========================================
@router.post("/{agent_id}", response_model=PolicyResponse)
def create_policy(
        agent_id: str,
        policy: PolicyCreate,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """Create a new policy with AI analysis"""

    # Verify agent
    agent = db.query(Agent).filter(Agent.id == agent_id, Agent.user_id == current_user.id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # AI analysis
    analysis = analyze_policy(policy.content)

    new_policy = Policy(
        agent_id=agent_id,
        title=policy.title,
        content=policy.content,
        category=policy.category or analysis.get("category", "general"),
        priority=policy.priority,
        analyzed_summary=analysis.get("summary")
    )

    db.add(new_policy)
    db.commit()
    db.refresh(new_policy)

    return new_policy


@router.post("/{agent_id}/upload", response_model=PolicyResponse)
async def upload_policy_file(
        agent_id: str,
        file: UploadFile = File(...),
        title: str = Form(...),
        category: Optional[str] = Form(None),
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """Upload a file with policy/rules"""

    # Verify agent
    agent = db.query(Agent).filter(Agent.id == agent_id, Agent.user_id == current_user.id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Validate file
    allowed = ["pdf", "docx", "doc", "txt"]
    file_ext = file.filename.split(".")[-1].lower()

    if file_ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Allowed: {allowed}")

    try:
        # Upload to Supabase
        content = await file.read()
        storage_path = f"policies/{current_user.id}/{agent_id}/{uuid.uuid4()}.{file_ext}"
        supabase.storage.from_("policy-files").upload(storage_path, content)

        # Download and extract text
        file_data = supabase.storage.from_("policy-files").download(storage_path)

        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp:
            tmp.write(file_data)
            tmp_path = tmp.name

        text = extract_text(tmp_path, file_ext)
        os.remove(tmp_path)

        # AI analysis
        analysis = analyze_policy(text)

        new_policy = Policy(
            agent_id=agent_id,
            title=title,
            content=text,
            category=category or analysis.get("category", "general"),
            analyzed_summary=analysis.get("summary")
        )

        db.add(new_policy)
        db.commit()
        db.refresh(new_policy)

        return new_policy

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{agent_id}", response_model=List[PolicyResponse])
def get_policies(
        agent_id: str,
        category: Optional[str] = None,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """Get all policies for an agent"""

    agent = db.query(Agent).filter(Agent.id == agent_id, Agent.user_id == current_user.id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    query = db.query(Policy).filter(Policy.agent_id == agent_id, Policy.is_active == True)

    if category:
        query = query.filter(Policy.category == category)

    return query.order_by(Policy.priority.desc()).all()


@router.put("/{agent_id}/{policy_id}", response_model=PolicyResponse)
def update_policy(
        agent_id: str,
        policy_id: str,
        policy: PolicyCreate,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """Update a policy"""

    existing = db.query(Policy).filter(Policy.id == policy_id, Policy.agent_id == agent_id).first()
    if not existing:
        raise HTTPException(status_code=404, detail="Policy not found")

    # Re-analyze
    analysis = analyze_policy(policy.content)

    existing.title = policy.title
    existing.content = policy.content
    existing.category = policy.category or analysis.get("category")
    existing.priority = policy.priority
    existing.analyzed_summary = analysis.get("summary")

    db.commit()
    db.refresh(existing)

    return existing


@router.patch("/{agent_id}/{policy_id}/toggle")
def toggle_policy(
        agent_id: str,
        policy_id: str,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """Toggle policy active/inactive"""

    policy = db.query(Policy).filter(Policy.id == policy_id, Policy.agent_id == agent_id).first()
    if not policy:
        raise HTTPException(status_code=404, detail="Policy not found")

    policy.is_active = not policy.is_active
    db.commit()

    return {"id": policy.id, "is_active": policy.is_active}


@router.delete("/{agent_id}/{policy_id}")
def delete_policy(
        agent_id: str,
        policy_id: str,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """Delete a policy"""

    policy = db.query(Policy).filter(Policy.id == policy_id, Policy.agent_id == agent_id).first()
    if not policy:
        raise HTTPException(status_code=404, detail="Policy not found")

    db.delete(policy)
    db.commit()

    return {"message": "Deleted successfully"}