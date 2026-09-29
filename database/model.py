from datetime import datetime, timezone

from sqlalchemy import (
    String,
    Text,
    DateTime,
    ForeignKey,
    Integer,
)

from sqlalchemy.orm import (
    Mapped,
    mapped_column,
    relationship,
)

from database.db import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer,primary_key=True,index=True)

    username: Mapped[str] = mapped_column(String(100),unique=True,nullable=False)

    email: Mapped[str] = mapped_column(String(255),unique=True,nullable=False,index=True)

    hashed_password: Mapped[str] = mapped_column(String(255),nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),default=lambda: datetime.now(timezone.utc))

    conversations: Mapped[list["Conversation"]] = relationship(back_populates="user",cascade="all, delete-orphan")


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(Integer,primary_key=True,index=True)

    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"),nullable=False,index=True)

    title: Mapped[str] = mapped_column(String(255),default="New Chat")

    tool_mode: Mapped[str] = mapped_column(String(100),default="auto")

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),default=lambda: datetime.now(timezone.utc))

    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),default=lambda: datetime.now(timezone.utc),onupdate=lambda: datetime.now(timezone.utc))

    user: Mapped["User"] = relationship(back_populates="conversations")

    messages: Mapped[list["Message"]] = relationship(back_populates="conversation",cascade="all, delete-orphan",order_by="Message.created_at")


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer,primary_key=True,index=True)

    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversations.id"),nullable=False,index=True)

    role: Mapped[str] = mapped_column(String(50),nullable=False)

    content: Mapped[str] = mapped_column(Text,nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),default=lambda: datetime.now(timezone.utc))

    conversation: Mapped["Conversation"] = relationship(back_populates="messages")