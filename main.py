from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
import tempfile

from dotenv import load_dotenv

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    Form,
    Depends,
    HTTPException,
)
from fastapi.middleware.cors import CORSMiddleware

from fastapi.responses import StreamingResponse

from pydantic import BaseModel, Field

from sqlalchemy.orm import Session

from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

from database.db import Base, engine, get_db
from database.model import User, Conversation, Message

from auth import (
    router as auth_router,
    get_current_user,
)

from tools.shuttle_tool import (
    predict_image,
    predict_video,
)

from tools.cifar_tool import predict_cifar10

import json


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# DATABASE / LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    Base.metadata.create_all(bind=engine)

    yield


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    lifespan=lifespan,
    title="AI Agent",
    description="Authenticated AI Agent with persistent conversations",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


import uuid
from fastapi.staticfiles import StaticFiles

# Mount static files
runs_dir = Path(__file__).resolve().parent / "runs"
runs_dir.mkdir(parents=True, exist_ok=True)
app.mount("/runs", StaticFiles(directory=str(runs_dir)), name="runs")

uploads_dir = Path(__file__).resolve().parent / "uploads"
uploads_dir.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(uploads_dir)), name="uploads")


# ============================================================
# AUTH ROUTER
# ============================================================

app.include_router(auth_router)


# ============================================================
# REQUEST SCHEMA
# ============================================================

class Input(BaseModel):
    question: str = Field(
        ...,
        title="Your question",
        description="Type your question here",
        examples=[
            "Who am I?",
            "What is a black hole?",
            "Explain LangChain",
        ],
    )


# ============================================================
# LLM CONFIGURATION
# ============================================================

groq = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.9,
)


gemini = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0.8,
)


# Groq is primary.
# Gemini is used if Groq fails.
llm = groq.with_fallbacks(
    [gemini]
)


# ============================================================
# AI AGENT
# ============================================================

agent = create_agent(
    model=llm,

    tools=[
        predict_image,
        predict_video,
        predict_cifar10,
    ],

    system_prompt="""
You are an AI assistant for an engineering student.

You can answer normal questions conversationally.

You also have access to machine-learning tools:

1. predict_image
   - Detect badminton shuttles in an image using YOLO.

2. predict_video
   - Track badminton shuttles in a video using YOLO.

3. predict_cifar10
   - Classify an image using a CIFAR-10 model.

Use an ML tool whenever the user asks you to actually
analyze an image or video.

Do not claim that you analyzed a file unless you
actually used the appropriate tool.

When a file path is provided in the user's message,
use the appropriate ML tool if the user's request
requires analysis of that file.
When a tool returns a result, always include the file path or URL (e.g. video_url or file_path starting with /runs/...) explicitly in your final response text so the user can preview the video or image output.
""",
)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
async def root():

    return {
        "message": "AI Agent API is running"
    }


# ============================================================
# CREATE CONVERSATION
# ============================================================

@app.post("/conversations")
async def create_conversation(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Create a new conversation for the authenticated user.
    """

    conversation = Conversation(
        user_id=current_user.id,
        title="New Chat",
        tool_mode="auto",
    )

    db.add(conversation)
    db.commit()
    db.refresh(conversation)

    return {
        "id": conversation.id,
        "title": conversation.title,
        "tool_mode": conversation.tool_mode,
        "created_at": conversation.created_at,
        "updated_at": conversation.updated_at,
    }


# ============================================================
# GET USER'S CONVERSATIONS
# ============================================================

@app.get("/conversations")
async def get_conversations(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Return all conversations belonging to the
    authenticated user.
    """

    conversations = (
        db.query(Conversation)
        .filter(
            Conversation.user_id == current_user.id
        )
        .order_by(
            Conversation.updated_at.desc()
        )
        .all()
    )

    return conversations


# ============================================================
# GET CONVERSATION MESSAGES
# ============================================================

@app.get("/conversations/{conversation_id}/messages")
async def get_conversation_messages(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Return messages from a conversation.

    The conversation must belong to the authenticated user.
    """

    conversation = (
        db.query(Conversation)
        .filter(
            Conversation.id == conversation_id,
            Conversation.user_id == current_user.id,
        )
        .first()
    )

    if conversation is None:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found",
        )

    messages = (
        db.query(Message)
        .filter(
            Message.conversation_id == conversation.id
        )
        .order_by(Message.created_at)
        .all()
    )

    return messages


# ============================================================
# HELPER:
# GET USER'S CONVERSATION
# ============================================================

def get_user_conversation(
    conversation_id: int,
    current_user: User,
    db: Session,
) -> Conversation:

    conversation = (
        db.query(Conversation)
        .filter(
            Conversation.id == conversation_id,
            Conversation.user_id == current_user.id,
        )
        .first()
    )

    if conversation is None:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found",
        )

    return conversation


# ============================================================
# HELPER:
# LOAD DATABASE HISTORY
# ============================================================

def load_conversation_messages(
    conversation_id: int,
    db: Session,
):
    """
    Convert database messages into the format expected
    by the LangChain agent.
    """

    messages = (
        db.query(Message)
        .filter(
            Message.conversation_id == conversation_id
        )
        .order_by(Message.created_at)
        .all()
    )

    agent_messages = []

    for message in messages:

        agent_messages.append(
            {
                "role": message.role,
                "content": message.content,
            }
        )

    return agent_messages


# ============================================================
# HELPER:
# SAVE MESSAGE
# ============================================================

def save_message(
    conversation_id: int,
    role: str,
    content,
    db: Session,
):
    """
    Save a user/assistant message to the database.

    LangChain messages can sometimes contain structured
    content such as dictionaries or lists, so normalize
    everything to a string before storing it.
    """

    if not isinstance(content, str):
        content = json.dumps(
            content,
            ensure_ascii=False,
            default=str,
        )

    message = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
    )

    db.add(message)

    return message
# ============================================================
# CHAT
# ============================================================

@app.post("/chat/{conversation_id}")
async def chat(
    data: Input,
    conversation_id: int,

    current_user: User = Depends(
        get_current_user
    ),

    db: Session = Depends(
        get_db
    ),
):
    """
    Send a message to an existing conversation.
    """

    # --------------------------------------------------------
    # 1. Verify conversation ownership
    # --------------------------------------------------------

    conversation = get_user_conversation(
        conversation_id=conversation_id,
        current_user=current_user,
        db=db,
    )

    # --------------------------------------------------------
    # 2. Save user message
    # --------------------------------------------------------

    save_message(
        conversation_id=conversation.id,
        role="user",
        content=data.question,
        db=db,
    )

    db.commit()

    # --------------------------------------------------------
    # 3. Load complete conversation history
    # --------------------------------------------------------

    agent_messages = load_conversation_messages(
        conversation_id=conversation.id,
        db=db,
    )

    # --------------------------------------------------------
    # 4. Run agent
    # --------------------------------------------------------

    formatted = await agent.ainvoke(
        {
            "messages": agent_messages
        }
    )

    # --------------------------------------------------------
    # 5. Extract final answer
    # --------------------------------------------------------

    answer = formatted["messages"][-1].content

    # --------------------------------------------------------
    # 6. Save assistant message
    # --------------------------------------------------------

    save_message(
        conversation_id=conversation.id,
        role="assistant",
        content=answer,
        db=db,
    )

    # --------------------------------------------------------
    # 7. Update conversation timestamp
    # --------------------------------------------------------

    conversation.updated_at = datetime.now(
        timezone.utc
    )

    # --------------------------------------------------------
    # 8. Commit everything
    # --------------------------------------------------------

    db.commit()

    # --------------------------------------------------------
    # 9. Return response
    # --------------------------------------------------------

    return {
        "conversation_id": conversation.id,
        "question": data.question,
        "answer": answer,
    }


# ============================================================
# STREAMING CHAT
# ============================================================

async def generate_response(
    question: str,
    conversation_id: int,
    current_user_id: int,
):
    """
    Stream agent response.

    A separate DB session is created because this function
    runs asynchronously while the response is being streamed.
    """

    db = next(get_db())

    try:

        # ----------------------------------------------------
        # Find user
        # ----------------------------------------------------

        user = (
            db.query(User)
            .filter(
                User.id == current_user_id
            )
            .first()
        )

        if user is None:
            return

        # ----------------------------------------------------
        # Verify conversation
        # ----------------------------------------------------

        conversation = (
            db.query(Conversation)
            .filter(
                Conversation.id == conversation_id,
                Conversation.user_id == user.id,
            )
            .first()
        )

        if conversation is None:
            return

        # ----------------------------------------------------
        # Save user message
        # ----------------------------------------------------

        save_message(
            conversation_id=conversation.id,
            role="user",
            content=question,
            db=db,
        )

        db.commit()

        # ----------------------------------------------------
        # Load history
        # ----------------------------------------------------

        agent_messages = load_conversation_messages(
            conversation_id=conversation.id,
            db=db,
        )

        # ----------------------------------------------------
        # Stream agent
        # ----------------------------------------------------

        full_response = ""

        async for chunk in agent.astream(
            {
                "messages": agent_messages
            },
            stream_mode="messages",
        ):

            message_chunk = chunk[0]

            content = getattr(
                message_chunk,
                "content",
                "",
            )

            if not content:
                continue

            full_response += content

            yield content

        # ----------------------------------------------------
        # Save final assistant response
        # ----------------------------------------------------

        if full_response:

            save_message(
                conversation_id=conversation.id,
                role="assistant",
                content=full_response,
                db=db,
            )

            conversation.updated_at = datetime.now(
                timezone.utc
            )

            db.commit()

    finally:
        db.close()


@app.post("/chat/stream/{conversation_id}")
async def chat_stream(
    data: Input,
    conversation_id: int,

    current_user: User = Depends(
        get_current_user
    ),

    db: Session = Depends(
        get_db
    ),
):
    """
    Stream an AI response from an authenticated
    conversation.
    """

    # --------------------------------------------------------
    # Verify ownership BEFORE starting the stream
    # --------------------------------------------------------

    get_user_conversation(
        conversation_id=conversation_id,
        current_user=current_user,
        db=db,
    )

    return StreamingResponse(
        generate_response(
            question=data.question,
            conversation_id=conversation_id,
            current_user_id=current_user.id,
        ),
        media_type="text/plain",
    )


# ============================================================
# FILE + CHAT
# ============================================================

@app.post("/upload-chat/{conversation_id}")
async def chat_with_file(
    conversation_id: int,

    file: UploadFile = File(...),

    question: str = Form(...),

    current_user: User = Depends(
        get_current_user
    ),

    db: Session = Depends(
        get_db
    ),
):
    """
    Upload a file and ask the AI agent to analyze it.
    """

    # --------------------------------------------------------
    # 1. Verify conversation ownership
    # --------------------------------------------------------

    conversation = get_user_conversation(
        conversation_id=conversation_id,
        current_user=current_user,
        db=db,
    )

    # --------------------------------------------------------
    # 2. Validate filename
    # --------------------------------------------------------

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Invalid file",
        )

    suffix = Path(
        file.filename
    ).suffix.lower()

    # --------------------------------------------------------
    # 3. Save temporary file
    # --------------------------------------------------------

    # --------------------------------------------------------
    # 3. Save uploaded file into uploads directory
    # --------------------------------------------------------

    file_id = uuid.uuid4().hex[:8]
    safe_filename = f"{file_id}_{file.filename}"
    saved_file_path = uploads_dir / safe_filename

    with open(saved_file_path, "wb") as f:
        f.write(await file.read())

    file_path = str(saved_file_path)
    web_url = f"/uploads/{safe_filename}"

    try:

        # ----------------------------------------------------
        # 4. Tell agent about uploaded file
        # ----------------------------------------------------

        augmented_question = f"""
{question}

The user uploaded a file.

Original filename: {file.filename}
File path for tool processing: {file_path}
Web accessible URL: {web_url}

Use the appropriate ML tool to analyze this file if the user's question requires analysis.
"""

        # ----------------------------------------------------
        # 5. Save user's question
        # ----------------------------------------------------

        save_message(
            conversation_id=conversation.id,
            role="user",
            content=question,
            db=db,
        )

        db.commit()

        # ----------------------------------------------------
        # 6. Load conversation history
        # ----------------------------------------------------

        agent_messages = load_conversation_messages(
            conversation_id=conversation.id,
            db=db,
        )

        # ----------------------------------------------------
        # 7. Add augmented file instruction
        # ----------------------------------------------------

        agent_messages[-1] = {
            "role": "user",
            "content": augmented_question,
        }

        # ----------------------------------------------------
        # 8. Run agent
        # ----------------------------------------------------

        formatted = await agent.ainvoke(
            {
                "messages": agent_messages
            }
        )

        # ----------------------------------------------------
        # 9. Extract response
        # ----------------------------------------------------

        answer = formatted["messages"][-1].content

        # ----------------------------------------------------
        # 10. Save assistant response
        # ----------------------------------------------------

        save_message(
            conversation_id=conversation.id,
            role="assistant",
            content=answer,
            db=db,
        )

        # ----------------------------------------------------
        # 11. Update conversation
        # ----------------------------------------------------

        conversation.updated_at = datetime.now(
            timezone.utc
        )

        db.commit()

        # ----------------------------------------------------
        # 12. Return
        # ----------------------------------------------------

        return {
            "conversation_id": conversation.id,
            "question": question,
            "filename": file.filename,
            "answer": answer,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )