import re
import sys
import os
from dataclasses import dataclass, field
from datetime import date
from typing import Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.database import db_connection
from core import config

logger = config.get_logger("api.services.caller_identity")

# Digits 5-6 of a roll number → candidate programme names. More than one
# candidate means ambiguous: resolve by asking, not by parsing. Code 01 is
# documented as "if it is 014 then ICT-CS else ICT", which reads a digit of the
# serial as part of the programme code — so it stays ambiguous here.
PROGRAM_CODES: dict[str, Tuple[str, ...]] = {
    "01": ("B Tech (ICT)", "B Tech (ICT-CS)"),
    "02": ("BE",),
    "03": ("B Tech (MnC)",),
    "04": ("B Tech (EVD)",),
    "11": ("M Tech (ICT)",),
    "12": ("MSc (IT)", "MS (IT)"),
    "13": ("MS (IT-Agri)", "MSc (ICT-ARD)"),
    "14": ("M Des (CD)", "M Des (IUXD)"),
    "15": ("M Tech (EC)", "M Tech (CS&ML)"),
    "16": ("M Tech (CS - Data Science)",),
    "17": ("M Tech (CS - Information Security)",),
    "18": ("MSc (DS)",),
    "19": ("MSc (AA)",),
    "21": ("Ph D",),
}

# Nominal length, used only to tell a current student from a graduate.
PROGRAM_DURATION_SEMESTERS: dict[str, int] = {
    "01": 8, "02": 8, "03": 8, "04": 8,
    "11": 4, "12": 4, "13": 4, "14": 4, "15": 4, "16": 4, "17": 4,
    "18": 4, "19": 4,
}

# 4-digit admission year + 2-digit programme code + 3-digit serial.
ROLL_NUMBER_RE = re.compile(r"^(\d{4})(\d{2})(\d{3})$")

EARLIEST_ADMISSION_YEAR = 2001


@dataclass(frozen=True)
class CallerIdentity:
    email: str
    role: str
    display_name: Optional[str] = None

    roll_number: Optional[str] = None
    admission_year: Optional[int] = None
    program_code: Optional[str] = None
    program_candidates: Tuple[str, ...] = field(default_factory=tuple)
    semester_estimate: Optional[int] = None
    is_probably_alumnus: bool = False

    @property
    def is_student(self) -> bool:
        return self.role.startswith("Student")

    @property
    def program(self) -> Optional[str]:
        """The programme, only when unambiguous."""
        return self.program_candidates[0] if len(self.program_candidates) == 1 else None

    @property
    def program_is_ambiguous(self) -> bool:
        return len(self.program_candidates) > 1

    @property
    def program_is_unmapped(self) -> bool:
        return self.program_code is not None and not self.program_candidates


def parse_roll_number(local_part: str) -> Optional[Tuple[int, str, str]]:
    m = ROLL_NUMBER_RE.match(local_part.strip())
    if not m:
        return None

    year_s, code, serial = m.groups()
    year = int(year_s)
    if not (EARLIEST_ADMISSION_YEAR <= year <= date.today().year + 1):
        return None

    return year, code, serial


def current_academic_term(today: Optional[date] = None) -> Tuple[int, int]:
    today = today or date.today()
    if today.month >= 7:
        return today.year, 0
    return today.year - 1, 1


def estimate_semester(admission_year: int, today: Optional[date] = None) -> int:
    academic_year_start, term_index = current_academic_term(today)
    return (academic_year_start - admission_year) * 2 + 1 + term_index


def _lookup_directory_name(email: str) -> Optional[str]:
    try:
        with db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT name FROM faculty WHERE email = %s LIMIT 1", (email,))
                row = cur.fetchone()
                if row:
                    return row[0]

                cur.execute("SELECT name FROM staff WHERE email = %s LIMIT 1", (email,))
                row = cur.fetchone()
                if row:
                    return row[0]
    except Exception as e:
        logger.error(f"Directory name lookup failed for {email}: {e}")

    return None


def resolve_caller(email: str, role: str, today: Optional[date] = None) -> CallerIdentity:
    local_part = email.split("@")[0]

    if not role.startswith("Student"):
        return CallerIdentity(
            email=email,
            role=role,
            display_name=_lookup_directory_name(email),
        )

    parsed = parse_roll_number(local_part)
    if not parsed:
        return CallerIdentity(email=email, role=role)

    admission_year, code, _serial = parsed
    candidates = PROGRAM_CODES.get(code, ())
    semester = estimate_semester(admission_year, today)
    duration = PROGRAM_DURATION_SEMESTERS.get(code)

    return CallerIdentity(
        email=email,
        role=role,
        roll_number=local_part,
        admission_year=admission_year,
        program_code=code,
        program_candidates=candidates,
        semester_estimate=semester,
        is_probably_alumnus=duration is not None and semester > duration,
    )
