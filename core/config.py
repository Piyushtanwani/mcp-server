import os
import logging
from typing import Dict, Any
from dotenv import load_dotenv

# Load .env from the project root (one level up from core/).
# override=False: real environment variables (systemd/docker/k8s) take
# precedence over .env, which is a local-development fallback only.
_env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
load_dotenv(dotenv_path=_env_path, override=False)

# ==============================================================================
# Logging Configuration
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger using the standard logging configuration."""
    return logging.getLogger(name)


logger = get_logger("core.config")


# ==============================================================================
# Database Configuration
# ==============================================================================
def get_db_config() -> Dict[str, Any]:
    """
    Retrieve and validate PostgreSQL connection parameters from environment variables.
    Raises ValueError if any required variable is missing.
    """
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT", "5432")
    name = os.getenv("DB_NAME")
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")

    missing = []
    if not host:     missing.append("DB_HOST")
    if not name:     missing.append("DB_NAME")
    if not user:     missing.append("DB_USER")
    if not password: missing.append("DB_PASSWORD")

    if missing:
        err = f"Missing required environment variable(s): {', '.join(missing)}"
        logger.error(err)
        raise ValueError(err)

    return {
        "host": host,
        "port": int(port),
        "database": name,
        "user": user,
        "password": password,
        "sslmode": os.getenv("DB_SSLMODE", "prefer"),
    }


# ==============================================================================
# External API Keys
# ==============================================================================
def get_gemini_api_key() -> str:
    """Return the configured Gemini API key, or empty string if not set."""
    return os.getenv("GEMINI_API_KEY", "")


def get_gemini_model() -> str:
    return os.getenv("GEMINI_MODEL", "gemini-flash-latest")


# ==============================================================================
# Retrieval and RAG Configuration
# ==============================================================================
def get_retrieval_limit() -> int:
    """Maximum number of records to retrieve per entity type during search."""
    try:
        return int(os.getenv("RETRIEVAL_LIMIT", "5"))
    except ValueError:
        return 5

def get_max_context_records() -> int:
    """Maximum total number of records allowed in the LLM context prompt."""
    try:
        return int(os.getenv("MAX_CONTEXT_RECORDS", "10"))
    except ValueError:
        return 10


# ==============================================================================
# Library / OPAC Configuration
# ==============================================================================
def get_library_opac_base_url() -> str:
    """Base URL of the DA-IICT Koha OPAC (no trailing slash)."""
    return os.getenv("LIBRARY_OPAC_BASE_URL", "https://opac.daiict.ac.in").rstrip("/")

# ==============================================================================
# Feedback Configuration
# ==============================================================================
def get_feedback_recipient_emails() -> list[str]:
    """Return a list of feedback recipient emails."""
    emails = os.getenv("FEEDBACK_RECIPIENT_EMAILS", "")
    return [e.strip() for e in emails.split(",") if e.strip()]

# ==============================================================================
# Authentication Configuration
# ==============================================================================
def get_google_client_id() -> str:
    """Return the Google Client ID for OAuth verification."""
    return os.getenv("GOOGLE_CLIENT_ID", "")
