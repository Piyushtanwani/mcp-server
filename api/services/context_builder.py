"""
Context Builder Service
=======================
Transforms raw database rows/dictionaries into clean, token-efficient
structured text for the Gemini RAG prompt.
"""
from typing import List, Dict, Any

from api.services.caller_identity import CallerIdentity


def build_caller_context(identity: CallerIdentity) -> str:
    out = [
        "**CALLER CONTEXT**",
        "You are speaking with a signed-in user whose identity was verified from "
        "their login credential. Resolve \"I\", \"me\" and \"my\" to this person, and "
        "use these details instead of asking for them.",
        f"Role: {identity.role}",
    ]

    if identity.display_name:
        out.append(f"Name: {identity.display_name}")
        out.append(
            f"For their own schedule, location or free time, call the timetable "
            f"tools with \"{identity.display_name}\"."
        )
    out.append(f"Email: {identity.email}")

    if not identity.is_student:
        return "\n".join(out)

    if identity.roll_number:
        out.append(f"Roll number: {identity.roll_number}")
        out.append(f"Admitted: {identity.admission_year}")

    if identity.is_probably_alumnus:
        out.append(
            "Programme status: their programme's nominal length has already passed, "
            "so they have most likely graduated. Do not state a current semester or "
            "fetch a timetable for them — say the timetable covers current students "
            "and ask what they need."
        )
        return "\n".join(out)

    if identity.program:
        out.append(f"Programme: {identity.program} (derived from the roll number)")
    elif identity.program_is_ambiguous:
        out.append(
            "Programme: UNKNOWN — the roll number narrows it to "
            + " or ".join(identity.program_candidates)
            + ". Ask which one before answering anything programme-specific."
        )
    elif identity.program_is_unmapped:
        out.append(
            f"Programme: UNKNOWN — roll number code {identity.program_code} is not "
            "mapped to a programme. Ask which programme they are in; never guess."
        )
    else:
        out.append("Programme: UNKNOWN — ask which programme they are in.")

    if identity.semester_estimate is not None:
        out.append(
            f"Likely semester: {identity.semester_estimate} — an estimate from the "
            "admission year, not a record. Use it as a default and let them correct it."
        )

    out.append(
        "Their elective choices are not known. A programme timetable is the core "
        "timetable only, so say that when you give them one."
    )

    return "\n".join(out)


def build_faculty_context(records: List[Dict[str, Any]]) -> str:
    """Format a list of faculty dicts into a structured string."""
    if not records:
        return "No relevant faculty records found for the query."

    out = ["FACULTY RESULTS\n"]
    for i, rec in enumerate(records, 1):
        out.append(f"{i}.")
        out.append(f"Name: {rec.get('name', 'N/A')}")
        if rec.get('faculty_type'):
            out.append(f"Type: {rec.get('faculty_type')}")
        if rec.get('specialization'):
            out.append(f"Specialization: {rec.get('specialization')}")
        if rec.get('education'):
            out.append(f"Education: {rec.get('education')}")
        if rec.get('email'):
            out.append(f"Email: {rec.get('email')}")
        if rec.get('phone'):
            out.append(f"Phone: {rec.get('phone')}")
        if rec.get('address'):
            out.append(f"Office: {rec.get('address')}")
        if rec.get('profile_url'):
            out.append(f"Profile: {rec.get('profile_url')}")
        out.append("") # Empty line between records
        
    return "\n".join(out)


def build_staff_context(records: List[Dict[str, Any]]) -> str:
    """Format a list of staff dicts into a structured string."""
    if not records:
        return "No relevant staff records found for the query."

    out = ["STAFF RESULTS\n"]
    for i, rec in enumerate(records, 1):
        out.append(f"{i}.")
        out.append(f"Name: {rec.get('name', 'N/A')}")
        if rec.get('designation'):
            out.append(f"Designation: {rec.get('designation')}")
        if rec.get('qualification'):
            out.append(f"Qualification: {rec.get('qualification')}")
        if rec.get('email'):
            out.append(f"Email: {rec.get('email')}")
        if rec.get('phone'):
            out.append(f"Phone: {rec.get('phone')}")
        if rec.get('address'):
            out.append(f"Office: {rec.get('address')}")
        if rec.get('profile_url'):
            out.append(f"Profile: {rec.get('profile_url')}")
        out.append("") # Empty line between records
        
    return "\n".join(out)
