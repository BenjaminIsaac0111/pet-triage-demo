"""The lexical baseline is only a fair comparison if its rules do what they say."""

from triage.baselines import lexical_triage, match_complaint


def test_matches_synonym():
    assert match_complaint("She keeps limping on her back leg") == "DEMO-010"


def test_more_matching_phrases_wins():
    # "scratching" is Itching; "red patch" and "sore" are both Skin lesion.
    assert match_complaint("A sore red patch he keeps scratching") == "DEMO-005"


def test_no_match_gives_none():
    assert match_complaint("Something is not right") is None


def test_emergency_phrase_skips_answers_and_sets_emergency():
    record = lexical_triage("He keeps dry heaving and nothing comes up", "He is limping too")
    assert record.urgency_tier == "EMERGENCY"
    assert record.complaint_code == "DEMO-002"  # the answers were not used


def test_no_flag_is_routine_and_uses_answers():
    record = lexical_triage("My dog is not quite right", "He has runny poo")
    assert record.urgency_tier == "ROUTINE"
    assert record.complaint_code == "DEMO-009"


def test_flag_in_answers_still_counts():
    record = lexical_triage("My dog is off colour", "This morning he collapsed")
    assert record.urgency_tier == "EMERGENCY"


def test_photo_for_visible_problems_only():
    assert lexical_triage("There's a cut on his paw").requires_photo_upload
    assert not lexical_triage("She has been sneezing all day").requires_photo_upload
