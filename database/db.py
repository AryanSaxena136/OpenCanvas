import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set")


class Base(DeclarativeBase):
    pass


# Use SQLite if MySQL is not available or if configured
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        pool_pre_ping=True,
    )
else:
    try:
        engine = create_engine(
            DATABASE_URL,
            pool_pre_ping=True,
        )
        # Test connection
        with engine.connect() as conn:
            pass
    except Exception:
        # Fallback to SQLite if MySQL/External DB is down
        sqlite_url = "sqlite:///./app.db"
        engine = create_engine(
            sqlite_url,
            connect_args={"check_same_thread": False},
            pool_pre_ping=True,
        )


SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)


def get_db():
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()