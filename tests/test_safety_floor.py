"""The safety floor must never lower urgency, and must only lift a flag with real evidence."""

from triage.safety_floor import apply_safety_floor, find_red_flags, quote_appears_in
from triage.schemas import RedFlagCheck


def check(flag_id, status, quote):
    return RedFlagCheck(flag_id=flag_id, status=status, evidence_quote=quote)


class TestFindRedFlags:
    def test_finds_emergency_phrase(self):
        assert find_red_flags("He keeps retching but nothing comes up") == ["bloat"]

    def test_ignores_case_and_punctuation(self):
        assert "breathing" in find_red_flags("She's STRUGGLING to breathe!!")

    def test_matches_whole_words_only(self):
        # "fitting" is a seizure phrase; "outfitting" should not trigger it.
        assert find_red_flags("We were outfitting the dog bed") == []

    def test_no_flags_for_routine_text(self):
        assert find_red_flags("A bit of yellow foam in the morning, otherwise fine") == []

    def test_still_matches_when_negated(self):
        # Matching words can't see "not"; reading context is the model's job.
        assert find_red_flags("He's not struggling to breathe") == ["breathing"]

    def test_pet_named_lily_is_not_a_poisoning(self):
        # Found in the web demo: "my cat Lily ... sneezing" was triaged as EMERGENCY.
        assert find_red_flags("My cat Lily has been sneezing since yesterday") == []

    def test_lily_plant_is_still_a_poisoning(self):
        assert find_red_flags("Lily chewed a lily from the vase") == ["poisoning"]
        assert find_red_flags("There's lily pollen on her face") == ["poisoning"]
        assert find_red_flags("She was sniffing the lilies") == ["poisoning"]


class TestQuoteAppearsIn:
    def test_exact_quote(self):
        assert quote_appears_in("not struggling to breathe", "He's not struggling to breathe")

    def test_light_paraphrase_accepted(self):
        assert quote_appears_in(
            "he is not struggling to breathe", "He's not struggling to breathe, he is fine"
        )

    def test_invented_quote_rejected(self):
        assert not quote_appears_in("breathing is completely normal", "He's sneezing a lot")

    def test_partly_invented_quote_rejected(self):
        # 3 of the 5 words are the owner's: below the 80% needed.
        assert not quote_appears_in("he is breathing totally normally", "He is breathing fast")

    def test_quote_exactly_at_the_threshold_accepted(self):
        # 4 of the 5 words are the owner's: exactly 80%.
        assert quote_appears_in("he keeps sneezing every morning", "He keeps sneezing every day")

    def test_single_word_rejected(self):
        assert not quote_appears_in("no", "no breathing problems")

    def test_two_word_quote_accepted(self):
        assert quote_appears_in("not coughing", "He's not coughing")

    def test_negation_moved_onto_the_sign_rejected(self):
        # Every word is in the owner's text, but the owner said the opposite.
        owner = "He is struggling to breathe and not eating since this morning."
        assert not quote_appears_in("he is not struggling to breathe", owner)

    def test_negation_spelled_differently_accepted(self):
        owner = "She isn't struggling to breathe at all, she's just sneezing"
        assert quote_appears_in("not struggling to breathe at all", owner)

    def test_negation_at_end_of_quote_checked_against_word_before(self):
        assert quote_appears_in("breathing problems none", "breathing problems none at all")
        assert not quote_appears_in("breathing problems none", "none, breathing problems")
        owner = "None of that, the dog has breathing problems"
        assert not quote_appears_in("the dog has breathing problems none", owner)


class TestApplySafetyFloor:
    OWNER = "He's not struggling to breathe, just sneezing since yesterday."

    def test_no_flags_keeps_model_urgency(self):
        result = apply_safety_floor("URGENT_24H", [], [], self.OWNER)
        assert result.urgency_tier == "URGENT_24H"
        assert not result.floor_raised_urgency

    def test_present_flag_raises_to_emergency(self):
        checks = [check("breathing", "present", "struggling to breathe")]
        result = apply_safety_floor("ROUTINE", ["breathing"], checks, "He's struggling to breathe")
        assert result.urgency_tier == "EMERGENCY"
        assert result.floor_raised_urgency

    def test_unclear_flag_raises_to_emergency(self):
        checks = [check("breathing", "unclear", "not struggling to breathe")]
        result = apply_safety_floor("ROUTINE", ["breathing"], checks, self.OWNER)
        assert result.urgency_tier == "EMERGENCY"

    def test_flag_the_model_ignored_raises_to_emergency(self):
        # The model didn't answer about the flag at all: doubt goes up.
        assert apply_safety_floor("ROUTINE", ["breathing"], [], self.OWNER).urgency_tier == (
            "EMERGENCY"
        )

    def test_absent_with_real_quote_lifts_flag(self):
        checks = [check("breathing", "absent", "not struggling to breathe")]
        result = apply_safety_floor("ROUTINE", ["breathing"], checks, self.OWNER)
        assert result.urgency_tier == "ROUTINE"
        assert result.flags_lifted == ["breathing"]

    def test_absent_with_made_up_quote_does_not_lift(self):
        checks = [check("breathing", "absent", "breathing is totally normal")]
        result = apply_safety_floor("ROUTINE", ["breathing"], checks, self.OWNER)
        assert result.urgency_tier == "EMERGENCY"

    def test_absent_with_reordered_negation_does_not_lift(self):
        owner = "He is struggling to breathe and not eating."
        checks = [check("breathing", "absent", "he is not struggling to breathe")]
        result = apply_safety_floor("ROUTINE", ["breathing"], checks, owner)
        assert result.urgency_tier == "EMERGENCY"
        assert result.flags_lifted == []

    def test_never_lowers_model_urgency(self):
        checks = [check("breathing", "absent", "not struggling to breathe")]
        result = apply_safety_floor("EMERGENCY", ["breathing"], checks, self.OWNER)
        assert result.urgency_tier == "EMERGENCY"
        assert not result.floor_raised_urgency

    def test_one_kept_flag_is_enough(self):
        checks = [
            check("breathing", "absent", "not struggling to breathe"),
            check("collapse", "present", "collapsed"),
        ]
        owner = "He's not struggling to breathe but he collapsed"
        result = apply_safety_floor("ROUTINE", ["breathing", "collapse"], checks, owner)
        assert result.urgency_tier == "EMERGENCY"
        assert result.flags_kept == ["collapse"]
