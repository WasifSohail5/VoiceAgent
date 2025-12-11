# voice.py
"""
OpenAI Realtime Voice API - WebSocket Handler
Real-time speech-to-speech conversation with AI Agent
"""

import os
import json
import base64
import asyncio
import uuid as uuid_module
from typing import Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException
from sqlalchemy.orm import Session

from main import (
    get_db, Agent, KnowledgeBase, Policy, Embedding,
    OPENAI_API_KEY
)
from knowledgebase import cosine_similarity
from openai import OpenAI

router = APIRouter(prefix="/voice", tags=["Voice - Realtime"])

OPENAI_REALTIME_URL = "wss://api.openai.com/v1/realtime? model=gpt-4o-realtime-preview-2024-12-17"

openai_client = OpenAI(api_key=OPENAI_API_KEY)


# ===========================================
# HELPER FUNCTIONS
# ===========================================
def get_policies_text(db: Session, agent_id) -> str:
    """Get all active policies as text"""
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


def get_knowledge_context(db: Session, agent_id) -> str:
    """Get summary of knowledge base for system prompt"""
    knowledge_items = db.query(KnowledgeBase).filter(
        KnowledgeBase.agent_id == agent_id,
        KnowledgeBase.status == "completed"
    ).all()

    if not knowledge_items:
        return "No knowledge base documents loaded."

    text = "## Available Knowledge Base:\n"
    for kb in knowledge_items:
        text += f"- {kb.filename} ({kb.chunk_count} chunks)\n"
    return text


def build_system_instructions(db: Session, agent: Agent) -> str:
    """Build complete system instructions for voice agent"""
    instructions = f"""You are {agent.name}, a professional voice AI assistant. 

## About You:
{agent.description or "You are a helpful customer service agent. "}

## Your Personality:
{agent.personality or "Professional, friendly, and helpful. "}

## Language:  
Primary language: {agent.language}
Speak naturally as this is a voice conversation.  Keep responses concise and conversational.

{get_policies_text(db, agent.id)}

{get_knowledge_context(db, agent.id)}

## Voice Conversation Guidelines:
1. Keep responses SHORT and conversational (this is voice, not text)
2. Use natural speech patterns
3. Confirm understanding before taking actions
4. If you don't know something, say so politely
5. Be warm, friendly, and professional
6. Avoid long lists - summarize instead
7. Ask clarifying questions when needed
"""
    return instructions


async def search_knowledge_for_voice(db: Session, agent_id, query: str, top_k: int = 3) -> str:
    """Search knowledge base and return context for voice response"""
    try:
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        query_embedding = response.data[0].embedding

        kb_ids = [kb.id for kb in db.query(KnowledgeBase).filter(
            KnowledgeBase.agent_id == agent_id,
            KnowledgeBase.status == "completed"
        ).all()]

        if not kb_ids:
            return ""

        embeddings = db.query(Embedding).filter(
            Embedding.knowledge_base_id.in_(kb_ids)
        ).all()

        results = []
        for emb in embeddings:
            sim = cosine_similarity(query_embedding, emb.embedding_vector)
            results.append({"text": emb.chunk_text, "sim": sim})

        results.sort(key=lambda x: x["sim"], reverse=True)
        top_results = results[:top_k]

        if not top_results:
            return ""

        context = "\n\nRelevant information:\n"
        for r in top_results:
            context += f"- {r['text'][: 300]}...\n"
        return context

    except Exception as e:
        print(f"Knowledge search error: {e}")
        return ""


# ===========================================
# WEBSOCKET CONNECTION MANAGER
# ===========================================
class VoiceConnectionManager:
    """Manages WebSocket connections for voice chat"""

    def __init__(self):
        self.active_connections: dict = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        print(f"✅ Voice session connected: {session_id}")

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            print(f"❌ Voice session disconnected: {session_id}")

    async def send_message(self, session_id: str, message: dict):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_json(message)


manager = VoiceConnectionManager()


# ===========================================
# MAIN VOICE WEBSOCKET ENDPOINT
# ===========================================
@router.websocket("/ws/{agent_id}")
async def voice_websocket_endpoint(
        websocket: WebSocket,
        agent_id: str,
        db: Session = Depends(get_db)
):
    """WebSocket endpoint for real-time voice conversation"""

    # Import here to avoid issues
    import websockets

    session_id = str(uuid_module.uuid4())

    # Get agent
    agent = db.query(Agent).filter(Agent.id == agent_id).first()

    if not agent:
        await websocket.close(code=4004, reason="Agent not found")
        return

    await manager.connect(websocket, session_id)

    system_instructions = build_system_instructions(db, agent)

    try:
        # Connect to OpenAI Realtime API using correct method
        openai_ws = await websockets.connect(
            OPENAI_REALTIME_URL,
            additional_headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "OpenAI-Beta": "realtime=v1"
            }
        )

        print(f"🔗 Connected to OpenAI Realtime API for session:  {session_id}")

        # Configure session
        session_config = {
            "type": "session. update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": system_instructions,
                "voice": agent.voice_id or "alloy",
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "input_audio_transcription": {
                    "model": "whisper-1"
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500
                },
                "temperature": 0.7,
                "max_response_output_tokens": 500
            }
        }

        await openai_ws.send(json.dumps(session_config))
        print(f"📤 Session configured for agent: {agent.name}")

        await websocket.send_json({
            "type": "session.ready",
            "agent_name": agent.name,
            "session_id": session_id
        })

        async def receive_from_client():
            try:
                while True:
                    data = await websocket.receive_json()

                    if data.get("type") == "audio. input":
                        audio_event = {
                            "type": "input_audio_buffer. append",
                            "audio": data.get("audio")
                        }
                        await openai_ws.send(json.dumps(audio_event))

                    elif data.get("type") == "audio.commit":
                        await openai_ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

                    elif data.get("type") == "audio.clear":
                        await openai_ws.send(json.dumps({"type": "input_audio_buffer.clear"}))

                    elif data.get("type") == "response.cancel":
                        await openai_ws.send(json.dumps({"type": "response.cancel"}))

                    elif data.get("type") == "text. input":
                        text_event = {
                            "type": "conversation.item.create",
                            "item": {
                                "type": "message",
                                "role": "user",
                                "content": [{"type": "input_text", "text": data.get("text")}]
                            }
                        }
                        await openai_ws.send(json.dumps(text_event))
                        await openai_ws.send(json.dumps({"type": "response.create"}))

            except WebSocketDisconnect:
                print(f"Client disconnected: {session_id}")
            except Exception as e:
                print(f"Client receive error: {e}")

        async def receive_from_openai():
            try:
                async for message in openai_ws:
                    event = json.loads(message)
                    event_type = event.get("type", "")

                    if event_type == "response.audio. delta":
                        await websocket.send_json({
                            "type": "audio.delta",
                            "audio": event.get("delta"),
                            "response_id": event.get("response_id")
                        })

                    elif event_type == "response.audio.done":
                        await websocket.send_json({
                            "type": "audio.done",
                            "response_id": event.get("response_id")
                        })

                    elif event_type == "conversation.item.input_audio_transcription. completed":
                        await websocket.send_json({
                            "type": "transcript.user",
                            "text": event.get("transcript", "")
                        })

                    elif event_type == "response.audio_transcript.delta":
                        await websocket.send_json({
                            "type": "transcript.agent.delta",
                            "text": event.get("delta", "")
                        })

                    elif event_type == "response.audio_transcript.done":
                        await websocket.send_json({
                            "type": "transcript.agent.done",
                            "text": event.get("transcript", "")
                        })

                    elif event_type == "response.done":
                        await websocket.send_json({"type": "response.done"})

                    elif event_type == "input_audio_buffer. speech_started":
                        await websocket.send_json({"type": "user.speaking. started"})

                    elif event_type == "input_audio_buffer.speech_stopped":
                        await websocket.send_json({"type": "user.speaking. stopped"})

                    elif event_type == "response.created":
                        await websocket.send_json({"type": "response.started"})

                    elif event_type == "error":
                        error_msg = event.get("error", {}).get("message", "Unknown error")
                        print(f"❌ OpenAI Error: {error_msg}")
                        await websocket.send_json({"type": "error", "message": error_msg})

                    elif event_type == "session.created":
                        print(f"✅ OpenAI session created")

                    elif event_type == "session.updated":
                        print(f"✅ OpenAI session updated")

            except Exception as e:
                print(f"OpenAI receive error:  {e}")

        await asyncio.gather(
            receive_from_client(),
            receive_from_openai()
        )

    except Exception as e:
        print(f"❌ Voice session error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass

    finally:
        manager.disconnect(session_id)
        try:
            await openai_ws.close()
        except:
            pass


# ===========================================
# REST ENDPOINTS
# ===========================================
@router.get("/voices")
def get_available_voices():
    """Get list of available voices"""
    return {
        "voices": [
            {"id": "alloy", "name": "Alloy", "description": "Neutral and balanced"},
            {"id": "ash", "name": "Ash", "description": "Soft and gentle"},
            {"id": "ballad", "name": "Ballad", "description": "Warm and engaging"},
            {"id": "coral", "name": "Coral", "description": "Clear and friendly"},
            {"id": "echo", "name": "Echo", "description": "Smooth and calm"},
            {"id": "sage", "name": "Sage", "description": "Wise and thoughtful"},
            {"id": "shimmer", "name": "Shimmer", "description": "Bright and energetic"},
            {"id": "verse", "name": "Verse", "description": "Dynamic and expressive"}
        ]
    }


@router.get("/test/{agent_id}")
def test_voice_setup(agent_id: str, db: Session = Depends(get_db)):
    """Test if agent is ready for voice chat"""
    agent = db.query(Agent).filter(Agent.id == agent_id).first()

    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    kb_count = db.query(KnowledgeBase).filter(
        KnowledgeBase.agent_id == agent_id,
        KnowledgeBase.status == "completed"
    ).count()

    policy_count = db.query(Policy).filter(
        Policy.agent_id == agent_id,
        Policy.is_active == True
    ).count()

    return {
        "ready": True,
        "agent": {
            "id": str(agent.id),
            "name": agent.name,
            "description": agent.description,
            "voice_id": agent.voice_id or "alloy"
        },
        "knowledge_base": {
            "documents": kb_count,
            "status": "ready" if kb_count > 0 else "empty"
        },
        "policies": {
            "count": policy_count,
            "status": "ready" if policy_count > 0 else "empty"
        },
        "websocket_url": f"/voice/ws/{agent_id}"
    }