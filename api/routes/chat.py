import os
import asyncio
from datetime import datetime
from fastapi import APIRouter, Request, HTTPException, Depends
from core import config
import hashlib
from core.database import db_connection
from core.rate_limit import limiter
from core.schemas import (
    ChatRequest, ChatResponse, ChatMessage,
    MAX_HISTORY_TURNS, MAX_HISTORY_CHARS, MAX_MESSAGE_CHARS,
)
from api.services import (
    call_gemini_api,
    process_fallback_message,
    clear_context_caches,
    is_gemini_available,
    record_gemini_failure,
    SYSTEM_INSTRUCTIONS_TEMPLATE,
)
from api.services.openai_service import (
    call_openai_api, is_openai_available, record_openai_failure
)
from api.auth import verify_google_token, resolve_role
from api.context import user_role_var, user_email_var
from api.services.caller_identity import resolve_caller
from api.services.context_builder import build_caller_context
from api.services.library_service import LibraryService

from scrapers import faculty_scraper, staff_scraper

logger = config.get_logger("api.routes.chat")
router = APIRouter()

# Shared library service instance (stateless, thread-safe)
_library_svc = LibraryService()

# ── Library intent keyword sets ───────────────────────────────────────────────
_LIBRARY_KEYWORDS = {
    "book", "books", "library", "resource centre", "resource center",
    "opac", "catalog", "catalogue", "borrow", "isbn", "publication",
    "textbook", "text book", "novel", "author", "publisher",
    "find a book", "search book", "check availability", "available book",
    "is it available", "copies available", "reserve book",
}


def _is_library_query(text: str) -> bool:
    """Return True if the message is asking about library/book resources."""
    t = text.lower()
    return any(kw in t for kw in _LIBRARY_KEYWORDS)


def _extract_book_query(text: str) -> str:
    """
    Strip conversational framing and return the core search keyword(s).

    Strategy
    --------
    Pass 1 — Preposition anchor:
        Look for phrases like "related to X", "about X", "on X", "called X".
        These reliably mark where the real search topic begins.

    Pass 2 — Preamble strip:
        Remove leading conversational filler layer by layer:
        "can u suggest me a book to read" → ""
        Then strip trailing noise like "from the resource centre".

    Examples
    --------
    "can u suggest me a book to read related to computer networks" → "computer networks"
    "is there any book related to fiction"                         → "fiction"
    "find me a book on machine learning"                           → "machine learning"
    "search for textbooks about databases"                         → "databases"
    "do you have anything on algorithms"                           → "algorithms"
    """
    import re

    _TRAILING_NOISE = re.compile(
        r"\s*(from\s+(the\s+)?resource\s+cent(re|er)"
        r"|from\s+(the\s+)?library"
        r"|in\s+(the\s+)?library"
        r"|from\s+opac"
        r"|from\s+catalog(ue)?"
        r"|please|thanks|thank\s+you)\s*",
        re.IGNORECASE,
    )

    def _clean_tail(s: str) -> str:
        return _TRAILING_NOISE.sub(" ", s).strip()

    # ── Pass 1: extract after a preposition anchor ─────────────────────────
    for pattern in [
        r"related\s+to\s+(.+)",
        r"(?<!\w)about\s+(.+)",
        r"(?:^|\s)on\s+(.+)",          # require space before 'on' to avoid 'novels'
        r"(?:^|\s)for\s+(.+)",         # same for 'for'
        r"(?<!\w)regarding\s+(.+)",
        r"(?<!\w)called\s+(.+)",
        r"(?<!\w)titled\s+(.+)",
        r"(?<!\w)named\s+(.+)",
        r"(?:^|\s)by\s+(.+)",          # "novels by Chetan Bhagat" → "Chetan Bhagat"
        r"(?:^|\s)of\s+(.+)",
    ]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            candidate = _clean_tail(m.group(1))
            # Must be a real topic, not another stop phrase
            if len(candidate) >= 2 and not re.fullmatch(
                r"(a\s+)?book(s)?|textbook(s)?|novel(s)?|any|one|it|them",
                candidate, re.IGNORECASE
            ):
                return candidate

    # ── Pass 2: preamble strip (layer by layer) ────────────────────────────
    cleaned = text.strip()

    # Remove leading conversational openers
    cleaned = re.sub(
        r"^(can\s+u|can\s+you|could\s+you|would\s+you|do\s+you|is\s+there|"
        r"are\s+there|do\s+we\s+have|suggest|recommend|help\s+me(\s+find)?)\s+",
        "", cleaned, flags=re.IGNORECASE,
    ).strip()

    # Remove "me a book / any book / some books / a textbook" with optional verb
    cleaned = re.sub(
        r"^(me\s+)?(find\s+|get\s+|show\s+)?"
        r"(a\s+|any\s+|some\s+|the\s+)?"
        r"(book|books|textbook|textbooks|novel|novels|copy|copies)\s*"
        r"(to\s+read\s+|to\s+borrow\s+|i\s+can\s+read\s+)?",
        "", cleaned, flags=re.IGNORECASE,
    ).strip()

    # Strip any remaining leading prepositions left over
    cleaned = re.sub(
        r"^(on|about|related\s+to|for|regarding|covering|dealing\s+with)\s+",
        "", cleaned, flags=re.IGNORECASE,
    ).strip()

    # Strip trailing noise
    cleaned = _clean_tail(cleaned)

    # If we didn't manage to strip anything meaningful, fall back to full text
    return cleaned if len(cleaned) >= 2 else text.strip()


async def handle_library_fallback(book_query: str) -> str:
    """Helper to process library fallback logic and return the formatted markdown string."""
    try:
        results = (await _library_svc.search_books(query=book_query, limit=5))["results"]
        if not results:
            return (
                f"📚 I searched the DA-IICT Resource Centre for **\"{book_query}\"** "
                f"but found no matching books.\n\n"
                "You can also search directly at: "
                "[opac.daiict.ac.in](https://opac.daiict.ac.in)"
            )

        lines = [
            f"📚 **Library Search Results for \"{ book_query }\"**",
            f"Found **{len(results)}** book(s) in the DA-IICT Resource Centre:",
            "",
        ]
        
        # Fetch availability in parallel
        async def fetch_avail(bib):
            if not bib: return None
            try:
                return await _library_svc.get_book_details(bib)
            except Exception:
                return None
                
        import asyncio
        details_list = await asyncio.gather(*(fetch_avail(b.get("biblionumber")) for b in results))

        for book, details in zip(results, details_list):
            title     = book.get("title", "Unknown Title")
            author    = book.get("author", "")
            link      = book.get("link", "")
            
            avail_str = "Unknown"
            if details:
                avail_str = f"{details.get('available_copies', 0)} / {details.get('total_copies', 0)}"

            lines.append(f"- **{title}**")
            if author:
                lines.append(f"  - Author: {author}")
            lines.append(f"  - Availability: {avail_str}")
            if link:
                lines.append(f"  - Link: {link}")
            lines.append("")

        return "\n".join(lines)

    except Exception as e:
        logger.error(f"Library search error in chat: {e}")
        return (
            f"⚠️ I tried to search the library catalog for **\"{book_query}\"** "
            f"but encountered an error: `{e}`\n\n"
            "Please try searching directly at: "
            "[opac.daiict.ac.in](https://opac.daiict.ac.in)"
        )


# ── Blocking-work budgets ─────────────────────────────────────────────────────
# Every LLM/scraper call below is synchronous. It is dispatched to a worker
# thread so it cannot block the event loop, and it is given a hard deadline so a
# single wedged request can never hold a connection open indefinitely. Keep
# LLM_TIMEOUT_S comfortably under the reverse proxy's read timeout so the client
# gets a real response instead of a dropped connection.
LLM_TIMEOUT_S = 45
SCRAPE_TIMEOUT_S = 120


async def _run_blocking(fn, *args, timeout: int = LLM_TIMEOUT_S):
    """
    Run a blocking callable off the event loop under a hard deadline.

    CAVEAT — the deadline abandons the call, it does not cancel it.
    `asyncio.to_thread` runs `fn` on a plain worker thread, and a thread cannot
    be interrupted from outside. On timeout this coroutine raises
    `TimeoutError` and the request gets an answer, but the thread keeps running
    to completion, still holding its DB connection and its outbound socket.

    Why that matters under load: a single chat can outlive its deadline by a
    lot. `gemini.MAX_TOOL_TURNS` (8) × `gemini._REQUEST_OPTIONS` (30s), plus
    tool dispatch time, allows roughly 240s of thread life against this 45s
    budget. Abandoned threads accumulate in the default executor
    (`min(32, cpu_count + 4)` workers); once every worker is occupied, new
    `to_thread` calls queue instead of starting, and latency climbs for
    everyone — a slower-motion version of the event-loop stall this helper
    exists to prevent.

    So if requests start timing out *in bulk* rather than individually, suspect
    executor saturation before suspecting the model or the network. The fix is
    to give the LLM clients an internal wall-clock budget matching
    LLM_TIMEOUT_S (so the thread stops when the caller does), or to run them on
    a dedicated bounded executor so saturation is explicit and observable.
    """
    return await asyncio.wait_for(asyncio.to_thread(fn, *args), timeout=timeout)


def sanitize_history(history) -> list[ChatMessage]:
    """
    Trim and sanitize client-supplied conversation history.

    The browser posts back whatever is in its localStorage, so history is
    untrusted input: a caller can forge assistant turns to walk the model past
    its instructions. Pydantic already coerces unknown senders to 'user'
    (core/schemas.py); here we additionally drop empties, cap per-message
    length, and keep only the most recent turns within a total character
    budget — which also bounds prompt cost and latency.
    """
    if not history:
        return []

    kept: list[ChatMessage] = []
    total = 0
    for msg in reversed(history):          # newest first, so caps drop old turns
        text = (msg.text or "").strip()
        if not text:
            continue
        text = text[:MAX_MESSAGE_CHARS]
        if len(kept) >= MAX_HISTORY_TURNS or total + len(text) > MAX_HISTORY_CHARS:
            break
        total += len(text)
        kept.append(ChatMessage(sender=msg.sender, text=text))

    return list(reversed(kept))


@limiter.limit("60/minute")
def _check_ip_auth_limit(request: Request):
    pass

def authenticate_request(request: Request) -> tuple[str, str]:
    """
    Authenticate the request and set request.state.email and request.state.role.
    Raises HTTPException(401) if missing or invalid credentials.
    Raises HTTPException(429) if rate limited.
    """
    _check_ip_auth_limit(request)
    
    auth = request.headers.get("Authorization")
    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    
    raw_key = auth.split(" ", 1)[1].strip()
    
    if raw_key.startswith("dau_sk_"):
        hashed_k = hashlib.sha256(raw_key.encode()).hexdigest()
        try:
            with db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT email FROM api_keys WHERE hashed_key = %s AND status = 'Active'", (hashed_k,))
                    row = cursor.fetchone()
                    if row:
                        email = row[0]
                        role = resolve_role(email)
                        request.state.email = email
                        request.state.role = role
                        return email, role
        except Exception:
            pass
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    email = verify_google_token(raw_key)
    role = resolve_role(email)
    request.state.email = email
    request.state.role = role
    return email, role

@router.post("/chat", response_model=ChatResponse)
@limiter.limit("20/minute")
async def chat_endpoint(request: Request, body: ChatRequest, auth: tuple[str, str] = Depends(authenticate_request)):
    """
    Main conversational endpoint.
    Routes between sync triggers, library search, Gemini RAG, and the local NLP fallback engine.
    """
    try:
        cleaned = body.message.strip().lower()
        history = sanitize_history(body.history)

        email, role = auth

        try:
            with db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "INSERT INTO mcp_analytics (tool_name, user_email, client_name) VALUES (%s, %s, %s)",
                        ('Web Chat', email, 'DAU Web Chat')
                    )
        except Exception as e:
            logger.error(f"Failed to log web chat analytics: {e}")

        # ── 0. Library Search Trigger (Fallback) ──────────────────────────────
        gemini_available = bool(os.getenv("GEMINI_API_KEY") and is_gemini_available())
        openai_available = bool(os.getenv("OPENAI_API_KEY") and is_openai_available())
        
        if _is_library_query(body.message) and not (gemini_available or openai_available):
            logger.info("Chat trigger: library search detected (No AI APIs available).")
            book_query = _extract_book_query(body.message)
            return ChatResponse(response=await handle_library_fallback(book_query))

        # ── 1. Sync Triggers ──────────────────────────────────────────────────
        if any(k in cleaned for k in ["sync staff", "scrape staff", "reload staff", "update staff database"]):
            logger.info("Chat trigger: manual staff synchronization initiated.")
            try:
                staff_data = await _run_blocking(
                    staff_scraper.scrape_staff_data, timeout=SCRAPE_TIMEOUT_S
                )
                if not staff_data:
                    return ChatResponse(response="[FAILED]: Could not scrape the live staff directory. Please check the logs.")
                await _run_blocking(
                    staff_scraper.save_to_database, staff_data, timeout=SCRAPE_TIMEOUT_S
                )
                clear_context_caches()
                return ChatResponse(response=(
                    f"**Staff Database synchronized successfully!**\n\n"
                    f"Reloaded **{len(staff_data)}** staff profiles from the live DA-IICT portal.\n"
                    "All query tools are now operating on the latest staff data!"
                ))
            except Exception as e:
                logger.error(f"Error during staff sync: {e}")
                return ChatResponse(response=f"[Error during staff synchronization]: {e}")

        elif any(k in cleaned for k in ["sync faculty", "sync faculties", "scrape faculty", "scrape faculties", "reload faculty", "reload faculties", "update faculty database"]):
            logger.info("Chat trigger: manual faculty synchronization initiated.")
            try:
                faculty_data = await _run_blocking(
                    faculty_scraper.scrape_faculty_data, timeout=SCRAPE_TIMEOUT_S
                )
                if not faculty_data:
                    return ChatResponse(response="[FAILED]: Could not scrape the live faculty directory. Please check the logs.")
                await _run_blocking(
                    faculty_scraper.save_to_database, faculty_data, timeout=SCRAPE_TIMEOUT_S
                )
                clear_context_caches()
                return ChatResponse(response=(
                    f"**Faculty Database synchronized successfully!**\n\n"
                    f"Reloaded **{len(faculty_data)}** faculty profiles from the live DA-IICT portal.\n"
                    "All query tools are now operating on the latest faculty data!"
                ))
            except Exception as e:
                logger.error(f"Error during faculty sync: {e}")
                return ChatResponse(response=f"[Error during faculty synchronization]: {e}")

        elif any(k in cleaned for k in ["sync", "scrape", "reload", "update database", "sync latest"]):
            logger.info("Chat trigger: full synchronization initiated.")
            try:
                faculty_data = await _run_blocking(
                    faculty_scraper.scrape_faculty_data, timeout=SCRAPE_TIMEOUT_S
                )
                if faculty_data:
                    await _run_blocking(
                        faculty_scraper.save_to_database, faculty_data, timeout=SCRAPE_TIMEOUT_S
                    )
                staff_data = await _run_blocking(
                    staff_scraper.scrape_staff_data, timeout=SCRAPE_TIMEOUT_S
                )
                if staff_data:
                    await _run_blocking(
                        staff_scraper.save_to_database, staff_data, timeout=SCRAPE_TIMEOUT_S
                    )
                clear_context_caches()
                return ChatResponse(response=(
                    f"**Full Database synchronized successfully!**\n\n"
                    f"- **{len(faculty_data) if faculty_data else 0}** faculty profiles\n"
                    f"- **{len(staff_data) if staff_data else 0}** staff profiles\n\n"
                    "All query tools are now operating on the latest university directory!"
                ))
            except Exception as e:
                logger.error(f"Error during full sync: {e}")
                return ChatResponse(response=f"[Error during synchronization]: {e}")

        # ── 2. Retrieval Strategy Selection ──────────────────────────────────────
        # (The old keyword bypass for "list all ..." queries is gone: it fired on
        # any query containing "list all" — e.g. "list all phd scholars in ML"
        # returned the faculty directory. list_faculty/list_staff/search_scholars
        # are bridged tools now; the model routes list queries correctly.)

        # Strategy A: Informational Queries (tool calling)
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")

        if (gemini_api_key and is_gemini_available()) or (openai_api_key and is_openai_available()):
            logger.info("Processing via tool-calling pipeline (Strategy A)...")
            # Directory data is no longer injected into the prompt — the model
            # reaches it through the bridged directory tools. The user's role is
            # published via contextvar so tool dispatch can redact contact
            # details for non-privileged users.
            user_role_var.set(request.state.role)
            user_email_var.set(request.state.email)

            identity = resolve_caller(request.state.email, request.state.role)
            now = datetime.now()

            system_instruction = SYSTEM_INSTRUCTIONS_TEMPLATE.format(
                current_date=now.strftime("%d %B %Y"),
                current_day=now.strftime("%A"),
                caller_context=build_caller_context(identity),
            )
            
            response_text = None

            # Both clients are synchronous: they are run in a worker thread under
            # a deadline so one slow call cannot stall the single uvicorn worker
            # (which would take the whole site down, not just this request).

            # Attempt 1: Gemini
            if gemini_api_key and is_gemini_available():
                try:
                    response_text, token_usage = await _run_blocking(
                        call_gemini_api, gemini_api_key, system_instruction, history
                    )
                    return ChatResponse(response=response_text)
                except asyncio.TimeoutError:
                    logger.error(f"Gemini RAG exceeded {LLM_TIMEOUT_S}s — abandoning.")
                    record_gemini_failure()
                except Exception:
                    logger.exception("Gemini RAG failed.")
                    record_gemini_failure()

            # Attempt 2: OpenAI Fallback
            if not response_text and openai_api_key and is_openai_available():
                try:
                    logger.info("Falling back to OpenAI RAG...")
                    response_text, token_usage = await _run_blocking(
                        call_openai_api, openai_api_key, system_instruction, history
                    )
                    return ChatResponse(response=response_text)
                except asyncio.TimeoutError:
                    logger.error(f"OpenAI RAG exceeded {LLM_TIMEOUT_S}s — abandoning.")
                    record_openai_failure()
                except Exception:
                    logger.exception("OpenAI RAG failed.")
                    record_openai_failure()
            
            # If both fail or are skipped, fall through to NLP fallback
            logger.warning("RAG engines unavailable or failed — falling back to local NLP engine/library.")
            if _is_library_query(body.message):
                return ChatResponse(response=await handle_library_fallback(_extract_book_query(body.message)))
            return ChatResponse(response=await _run_blocking(process_fallback_message, body.message))

        # ── 3. Local NLP Fallback ──────────────────────────────────────────────
        logger.info("No AI APIs available or in cooldown — using local NLP engine.")
        if _is_library_query(body.message):
            return ChatResponse(response=await handle_library_fallback(_extract_book_query(body.message)))
        return ChatResponse(response=await _run_blocking(process_fallback_message, body.message))

    except HTTPException:
        # Re-raise HTTPExceptions so FastAPI can return the correct status code (e.g. 401)
        raise
    except asyncio.TimeoutError:
        # A blocking stage blew its deadline. Answer the client rather than
        # letting the connection hang until the proxy resets it — a reset is
        # what surfaces in the browser as "Failed to fetch".
        logger.error("Chat request timed out waiting on a blocking stage.")
        return ChatResponse(response=(
            "⏳ That took longer than expected and I had to stop. "
            "Please try again, or ask a narrower question."
        ))
    except Exception as e:
        # Must return a ChatResponse: falling off the end returns None, which
        # fails response_model validation and turns every error into an opaque 500.
        # Answer the client instead of raising. This used to be
        # `raise HTTPException(500, detail=str(e))`, which put the raw exception
        # text in the response body and gave the chat UI nothing useful to show.
        logger.exception(f"Unhandled error in chat endpoint: {e}")
        return ChatResponse(response=(
            "⚠️ Something went wrong on my side while answering that. "
            "Please try again in a moment."
        ))