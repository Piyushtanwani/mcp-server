"""
Tests for caller identity resolution and the CALLER CONTEXT prompt block.

"""
from datetime import date
from unittest.mock import MagicMock

import pytest

from api.services import caller_identity, gemini
from api.services.caller_identity import (
    CallerIdentity,
    estimate_semester,
    parse_roll_number,
    resolve_caller,
)
from api.services.context_builder import build_caller_context

TODAY = date(2026, 8, 6)


def mock_directory(mocker, name):
    conn = MagicMock()
    cur = MagicMock()
    conn.__enter__.return_value = conn
    conn.cursor.return_value.__enter__.return_value = cur
    cur.fetchone.return_value = (name,) if name else None
    mocker.patch("api.services.caller_identity.db_connection", return_value=conn)
    return cur


class TestParseRollNumber:
    def test_valid_roll_number_splits_into_year_code_serial(self):
        assert parse_roll_number("202411034") == (2024, "11", "034")

    @pytest.mark.parametrize("bad", [
        "",
        "abc",
        "20241103",        # too short
        "2024110345",      # too long
        "20241103x",       # not all digits
        "199911034",       # admission year before the institute existed
    ])
    def test_malformed_input_returns_none_rather_than_a_partial_parse(self, bad):
        assert parse_roll_number(bad) is None

    def test_surrounding_whitespace_is_tolerated(self):
        assert parse_roll_number(" 202411034 ") == (2024, "11", "034")

    def test_future_admission_year_is_allowed_one_year_ahead(self):
        next_year = date.today().year + 1
        assert parse_roll_number(f"{next_year}11034") is not None


class TestSemesterEstimate:
    def test_odd_term_after_july(self):
        assert estimate_semester(2024, date(2026, 8, 6)) == 5

    def test_even_term_before_july_belongs_to_previous_academic_year(self):
        assert estimate_semester(2024, date(2027, 1, 10)) == 6

    def test_first_semester_in_admission_year(self):
        assert estimate_semester(2026, TODAY) == 1


class TestResolveCaller:
    def test_unambiguous_code_resolves_to_one_programme(self, mocker):
        i = resolve_caller("202511034@dau.ac.in", "Student", TODAY)

        assert i.program == "M Tech (ICT)"
        assert not i.program_is_ambiguous
        assert i.admission_year == 2025
        assert i.semester_estimate == 3
        assert not i.is_probably_alumnus

    def test_two_year_programme_is_alumnus_once_its_four_semesters_have_passed(self):
        """A 2024 M Tech admit has finished by Aug 2026 — a 4-year admit has not."""
        assert resolve_caller("202411034@dau.ac.in", "Student", TODAY).is_probably_alumnus
        assert not resolve_caller("202403034@dau.ac.in", "Student", TODAY).is_probably_alumnus

    def test_code_01_stays_ambiguous_and_is_never_resolved(self):
        """Splitting ICT from ICT-CS needs a digit of the serial. Ask instead."""
        i = resolve_caller("202401034@dau.ac.in", "Student", TODAY)

        assert i.program is None
        assert i.program_is_ambiguous
        assert i.program_candidates == ("B Tech (ICT)", "B Tech (ICT-CS)")

    def test_unmapped_code_yields_no_programme(self):
        i = resolve_caller("202499034@dau.ac.in", "Student", TODAY)

        assert i.program is None
        assert i.program_is_unmapped
        assert i.program_candidates == ()

    def test_malformed_roll_number_yields_identity_without_derived_fields(self):
        i = resolve_caller("someone@dau.ac.in", "Student", TODAY)

        assert i.email == "someone@dau.ac.in"
        assert i.roll_number is None
        assert i.program is None
        assert i.semester_estimate is None
        assert not i.program_is_unmapped

    def test_long_past_admission_is_flagged_as_alumnus(self):
        i = resolve_caller("201401034@dau.ac.in", "Student", TODAY)

        assert i.is_probably_alumnus

    def test_phd_has_no_duration_so_is_never_called_an_alumnus(self):
        i = resolve_caller("201421034@dau.ac.in", "Student", TODAY)

        assert not i.is_probably_alumnus

    def test_faculty_identity_carries_their_directory_name(self, mocker):
        mock_directory(mocker, "Ankush Chander")
        i = resolve_caller("ankush@dau.ac.in", "Faculty", TODAY)

        assert i.display_name == "Ankush Chander"
        assert not i.is_student
        assert i.roll_number is None

    def test_directory_failure_degrades_instead_of_raising(self, mocker):
        mocker.patch(
            "api.services.caller_identity.db_connection",
            side_effect=Exception("db down"),
        )
        i = resolve_caller("ankush@dau.ac.in", "Faculty", TODAY)

        assert i.display_name is None
        assert i.email == "ankush@dau.ac.in"

    def test_student_maintainer_is_still_treated_as_a_student(self):
        i = resolve_caller("202411034@dau.ac.in", "Student / Maintainer", TODAY)

        assert i.is_student
        assert i.program == "M Tech (ICT)"


class TestCallerContextBlock:
    def test_faculty_block_names_them_for_the_timetable_tools(self, mocker):
        mock_directory(mocker, "Ankush Chander")
        block = build_caller_context(resolve_caller("ankush@dau.ac.in", "Faculty", TODAY))

        assert "Ankush Chander" in block
        assert "timetable tools" in block

    def test_student_block_states_programme_semester_and_elective_caveat(self):
        block = build_caller_context(resolve_caller("202511034@dau.ac.in", "Student", TODAY))

        assert "M Tech (ICT)" in block
        assert "Likely semester: 3" in block
        assert "estimate" in block
        assert "elective" in block.lower()

    def test_ambiguous_programme_block_asks_and_offers_no_default(self):
        block = build_caller_context(resolve_caller("202401034@dau.ac.in", "Student", TODAY))

        assert "UNKNOWN" in block
        assert "B Tech (ICT)" in block and "B Tech (ICT-CS)" in block
        assert "Ask" in block

    def test_unmapped_programme_block_forbids_guessing(self):
        block = build_caller_context(resolve_caller("202499034@dau.ac.in", "Student", TODAY))

        assert "UNKNOWN" in block
        assert "never guess" in block.lower()

    def test_alumnus_block_states_no_semester(self):
        block = build_caller_context(resolve_caller("201401034@dau.ac.in", "Student", TODAY))

        assert "graduated" in block.lower()
        assert "Likely semester" not in block

    def test_block_never_leaks_a_programme_it_could_not_resolve(self):
        """An UNKNOWN programme must not also print a candidate as if it were the answer."""
        block = build_caller_context(
            CallerIdentity(email="x@dau.ac.in", role="Student", program_code="15")
        )

        assert "UNKNOWN" in block
        assert "M Tech (EC)" not in block


class TestIdentityGuardrails:
    def test_prompt_no_longer_claims_the_role_is_invisible(self):
        """That sentence stops being true the moment identity is injected."""
        assert "which you cannot see or change" not in gemini.SYSTEM_INSTRUCTIONS_TEMPLATE

    @pytest.mark.parametrize("clause", [
        "CALLER CONTEXT",
        "login credential",
        "claimed in a chat message",
    ])
    def test_prompt_separates_verified_identity_from_claimed_identity(self, clause):
        assert clause in gemini.SYSTEM_INSTRUCTIONS_TEMPLATE

    def test_template_renders_with_the_caller_block(self):
        rendered = gemini.SYSTEM_INSTRUCTIONS_TEMPLATE.format(
            current_date="06 August 2026",
            current_day="Thursday",
            caller_context=build_caller_context(
                CallerIdentity(email="202411034@dau.ac.in", role="Student")
            ),
        )

        assert "06 August 2026" in rendered
        assert "202411034@dau.ac.in" in rendered
