# app/agent.py
"""
Agent - Create agents, RAG responses, manage everything
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from datetime import datetime

from openai import OpenAI

from main import (
    get_db, get_current_user, User, Agent, KnowledgeBase, Policy, Embedding,
    OPENAI_API_KEY
)
from knowledgebase import cosine_similarity

router = APIRouter(prefix="/agents", tags=["Agents"])

openai_client = OpenAI(api_key=OPENAI_API_KEY)


# ===========================================
# SCHEMAS
# ===========================================
class AgentCreate(BaseModel):
    name: str
    description: Optional[str] = None
    voice_id: Optional[str] = None
    language: Optional[str] = "en"
    personality: Optional[str] = None


class AgentUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    voice_id: Optional[str] = None
    language: Optional[str] = None
    personality: Optional[str] = None
    is_active: Optional[bool] = None


class AgentResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    voice_id: Optional[str]
    language: str
    personality: Optional[str]
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


class ChatRequest(BaseModel):
    message: str
    conversation_history: List[dict] = []


class ChatResponse(BaseModel):
    response: str
    agent_id: str


# ===========================================
# HELPER FUNCTIONS
# ===========================================
def get_policies_prompt(db: Session, agent_id: str) -> str:
    """Format all policies for system prompt"""

    policies = db.query(Policy).filter(
        Policy.agent_id == agent_id,
        Policy.is_active == True
    ).order_by(Policy.priority.desc()).all()

    if not policies:
        return "No specific policies defined."

    text = "## Agent Policies & Rules:\n\n"
    for p in policies:
        text += f"### {p.title} ({p.category})\n{p.content}\n\n"

    return text


def build_system_prompt(db: Session, agent: Agent) -> str:
    """Build complete system prompt for agent"""

    prompt = f"""You are {agent.name}, a professional voice AI assistant. 

## About You:
{agent.description or "You are a helpful customer service agent. "}

## Your Personality:
{agent.personality or "Professional, friendly, and helpful."}

## Language: 
Primary language: {agent.language}

"""

    prompt += get_policies_prompt(db, agent.id)

    prompt += """
## Knowledge Base: 
Use the context provided from knowledge base to answer questions. 
If no relevant info found, acknowledge and offer alternatives.

## Guidelines:
1. Be concise - this is voice conversation
2. Confirm understanding before actions
3. Escalate to human when necessary
4. Always be polite and professional
"""

    return prompt


def search_knowledge(db: Session, agent_id: str, query: str, top_k: int = 3) -> str:
    """Search knowledge base and return context"""

    # Get query embedding
    response = openai_client.embeddings.create(model="text-embedding-3-small", input=query)
    query_embedding = response.data[0].embedding

    # Get all embeddings
    kb_ids = [kb.id for kb in db.query(KnowledgeBase).filter(
        KnowledgeBase.agent_id == agent_id,
        KnowledgeBase.status == "completed"
    ).all()]

    if not kb_ids:
        return ""

    embeddings = db.query(Embedding).filter(Embedding.knowledge_base_id.in_(kb_ids)).all()

    # Find similar
    results = []
    for emb in embeddings:
        sim = cosine_similarity(query_embedding, emb.embedding_vector)
        results.append({"text": emb.chunk_text, "sim": sim})

    results.sort(key=lambda x: x["sim"], reverse=True)
    top_results = results[:top_k]

    if not top_results:
        return ""

    context = "\n\n## Relevant Information:\n"
    for r in top_results:
        context += f"- {r['text'][: 500]}...\n"

    return context


# ===========================================
# API ROUTES
# ===========================================
@router.post("/", response_model=AgentResponse)
def create_agent(
        agent: AgentCreate,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """Create a new voice agent"""

    new_agent = Agent(
        user_id=current_user.id,
        name=agent.name,
        description=agent.description,
        voice_id=agent.voice_id,
        language=agent.language,
        personality=agent.personality
    )

    db.add(new_agent)
    db.commit()
    db.refresh(new_agent)

    return new_agent


@router.get("/", response_model=List[AgentResponse])
def list_agents(
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """List all agents for user"""

    return db.query(Agent).filter(Agent.user_id == current_user.id).all()


@router.get("/{agent_id}")
def get_agent(
        agent_id: str,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """Get agent with all details"""

    agent = db.query(Agent).filter(Agent.id == agent_id, Agent.user_id == current_user.id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    knowledge = db.query(KnowledgeBase).filter(KnowledgeBase.agent_id == agent_id).all()
    policies = db.query(Policy).filter(Policy.agent_id == agent_id).all()

    return {
        "agent": AgentResponse.model_validate(agent),
        "knowledge_base": knowledge,
        "policies": policies,
        "stats": {
            "total_documents": len(knowledge),
            "processed_documents": len([k for k in knowledge if k.status == "completed"]),
            "total_policies": len(policies),
            "active_policies": len([p for p in policies if p.is_active])
        }
    }


@router.put("/{agent_id}", response_model=AgentResponse)
def update_agent(
        agent_id: str,
        data: AgentUpdate,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """Update agent"""

    agent = db.query(Agent).filter(Agent.id == agent_id, Agent.user_id == current_user.id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    for key, value in data.model_dump(exclude_unset=True).items():
        if value is not None:
            setattr(agent, key, value)

    db.commit()
    db.refresh(agent)

    return agent


@router.delete("/{agent_id}")
def delete_agent(
        agent_id: str,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """Delete agent and all data"""

    agent = db.query(Agent).filter(Agent.id == agent_id, Agent.user_id == current_user.id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    db.delete(agent)
    db.commit()

    return {"message": "Agent deleted successfully"}


@router.post("/{agent_id}/chat", response_model=ChatResponse)
def chat_with_agent(
        agent_id: str,
        chat: ChatRequest,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """Chat with agent using RAG"""

    agent = db.query(Agent).filter(Agent.id == agent_id, Agent.user_id == current_user.id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Build system prompt
    system_prompt = build_system_prompt(db, agent)

    # Search knowledge base
    context = search_knowledge(db, agent_id, chat.message)

    # Build messages
    messages = [{"role": "system", "content": system_prompt + context}]

    if chat.conversation_history:
        messages.extend(chat.conversation_history)

    messages.append({"role": "user", "content": chat.message})

    # Generate response
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=500,
        temperature=0.7
    )

    return ChatResponse(
        response=response.choices[0].message.content,
        agent_id=agent_id
    )


@router.get("/{agent_id}/system-prompt")
def get_system_prompt(
        agent_id: str,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """Get system prompt for debugging"""

    agent = db.query(Agent).filter(Agent.id == agent_id, Agent.user_id == current_user.id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    return {"agent_id": agent_id, "system_prompt": build_system_prompt(db, agent)}