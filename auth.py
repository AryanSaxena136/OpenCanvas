import os
from datetime import datetime, timedelta, timezone

import jwt
from dotenv import load_dotenv

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)

from fastapi.security import (
    OAuth2PasswordBearer,
    OAuth2PasswordRequestForm,
)

from pwdlib import PasswordHash

from pydantic import (
    BaseModel,
    EmailStr,
    ConfigDict,
)

from sqlalchemy.orm import Session

from database.db import get_db
from database.model import User


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")

if not SECRET_KEY:
    raise RuntimeError(
        "SECRET_KEY is not set in the environment."
    )

ALGORITHM = "HS256"

ACCESS_TOKEN_EXPIRE_MINUTES = 30


# ============================================================
# ROUTER
# ============================================================

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
)


# ============================================================
# PASSWORD HASHING
# ============================================================

password_hash = PasswordHash.recommended()


def get_password_hash(password: str) -> str:
    """
    Convert a plain-text password into a secure Argon2 hash.
    """

    return password_hash.hash(password)


def verify_password(
    plain_password: str,
    hashed_password: str,
) -> bool:
    """
    Verify a plain-text password against its stored hash.
    """

    return password_hash.verify(
        plain_password,
        hashed_password,
    )


# ============================================================
# OAUTH2
# ============================================================

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/auth/login",
)


# ============================================================
# PYDANTIC SCHEMAS
# ============================================================

class UserCreate(BaseModel):
    """
    Data required when creating a new user.
    """

    username: str
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    """
    Public user information.

    hashed_password is intentionally NOT included.
    """

    id: int
    username: str
    email: EmailStr

    model_config = ConfigDict(
        from_attributes=True,
    )


class TokenResponse(BaseModel):
    """
    Response returned after successful login.
    """

    access_token: str
    token_type: str


# ============================================================
# JWT
# ============================================================

def create_access_token(
    data: dict,
    expires_delta: timedelta | None = None,
) -> str:
    """
    Create a signed JWT access token.
    """

    payload = data.copy()

    if expires_delta is None:
        expires_delta = timedelta(
            minutes=ACCESS_TOKEN_EXPIRE_MINUTES
        )

    expire = (
        datetime.now(timezone.utc)
        + expires_delta
    )

    payload.update(
        {
            "exp": expire,
        }
    )

    return jwt.encode(
        payload,
        SECRET_KEY,
        algorithm=ALGORITHM,
    )


# ============================================================
# REGISTER
# ============================================================

@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
)
def register_user(
    user: UserCreate,
    db: Session = Depends(get_db),
):
    """
    Register a new user.
    """

    # --------------------------------------------------------
    # Check username
    # --------------------------------------------------------

    existing_username = (
        db.query(User)
        .filter(User.username == user.username)
        .first()
    )

    if existing_username:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered",
        )

    # --------------------------------------------------------
    # Check email
    # --------------------------------------------------------

    existing_email = (
        db.query(User)
        .filter(User.email == user.email)
        .first()
    )

    if existing_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )

    # --------------------------------------------------------
    # Hash password
    # --------------------------------------------------------

    hashed_password = get_password_hash(
        user.password
    )

    # --------------------------------------------------------
    # Create database user
    # --------------------------------------------------------

    new_user = User(
        username=user.username,
        email=user.email,
        hashed_password=hashed_password,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return new_user


# ============================================================
# LOGIN
# ============================================================

@router.post(
    "/login",
    response_model=TokenResponse,
)
def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    """
    Authenticate a user and return an access token.
    """

    # --------------------------------------------------------
    # Find user
    # --------------------------------------------------------

    user = (
        db.query(User)
        .filter(User.username == form_data.username)
        .first()
    )

    # --------------------------------------------------------
    # Verify credentials
    # --------------------------------------------------------

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={
                "WWW-Authenticate": "Bearer",
            },
        )

    if not verify_password(
        form_data.password,
        user.hashed_password,
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={
                "WWW-Authenticate": "Bearer",
            },
        )

    # --------------------------------------------------------
    # Create JWT
    # --------------------------------------------------------

    access_token = create_access_token(
        data={
            "sub": str(user.id),
        },
        expires_delta=timedelta(
            minutes=ACCESS_TOKEN_EXPIRE_MINUTES
        ),
    )

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
    )


# ============================================================
# CURRENT USER
# ============================================================

def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    """
    Validate the JWT and return the authenticated user.
    """

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={
            "WWW-Authenticate": "Bearer",
        },
    )

    # --------------------------------------------------------
    # Decode JWT
    # --------------------------------------------------------

    try:

        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM],
        )

        user_id = payload.get("sub")

        if user_id is None:
            raise credentials_exception

        try:
            user_id = int(user_id)

        except (TypeError, ValueError):
            raise credentials_exception

    except jwt.PyJWTError:
        raise credentials_exception

    # --------------------------------------------------------
    # Find user
    # --------------------------------------------------------

    user = (
        db.query(User)
        .filter(User.id == user_id)
        .first()
    )

    if user is None:
        raise credentials_exception

    return user