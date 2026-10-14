from triage.red_flags import apply_floor, lift_agreement, mark_unclear, quote_supported, scan
from triage.schemas import RedFlagCheck


def check(flag_id, status, quote):
    return RedFlagCheck(flag_id=flag_id, status=status, evidence_quote=quote)


class TestScan:
    def test_finds_emergency_phrase(self):
        assert scan("He keeps retching but nothing comes up") == ["bloat"]

    def test_ignores_case_and_punctuation(self):
        assert "breathing" in scan("She's STRUGGLING to breathe!!")

    def test_matches_whole_words_only(self):
        # "fitting" is a seizure phrase; "outfitting" should not trigger it.
        assert scan("We were outfitting the dog bed") == []

    def test_no_flags_for_routine_text(self):
        assert scan("A bit of yellow foam in the morning, otherwise fine") == []

    def test_still_matches_when_negated(self):
        # Lexical matching cannot see "not"; that is the LLM's job.
        assert scan("He's not struggling to breathe") == ["breathing"]


class TestQuoteSupported:
    def test_exact_quote(self):
        assert quote_supported("not struggling to breathe", "He's not struggling to breathe at all")

    def test_light_paraphrase_accepted(self):
        assert quote_supported(
            "he is not struggling to breathe", "He's not struggling to breathe, he is fine"
        )

    def test_invented_quote_rejected(self):
        assert not quote_supported("breathing is completely normal", "He's sneezing a lot")

    def test_single_word_rejected(self):
        assert not quote_supported("no", "no breathing problems")


class TestApplyFloor:
    OWNER = "He's not struggling to breathe, just sneezing since yesterday."

    def test_no_flags_keeps_model_urgency(self):
        result = apply_floor("URGENT_24H", [], [], self.OWNER)
        assert result.urgency_tier == "URGENT_24H"
        assert not result.floor_applied

    def test_present_flag_raises_to_emergency(self):
        checks = [check("breathing", "present", "struggling to breathe")]
        result = apply_floor("ROUTINE", ["breathing"], checks, "He's struggling to breathe")
        assert result.urgency_tier == "EMERGENCY"
        assert result.floor_applied

    def test_unclear_flag_raises_to_emergency(self):
        checks = [check("breathing", "unclear", "not struggling to breathe")]
        assert apply_floor("ROUTINE", ["breathing"], checks, self.OWNER).urgency_tier == "EMERGENCY"

    def test_missing_check_raises_to_emergency(self):
        # The model ignored the flag entirely: doubt goes up.
        assert apply_floor("ROUTINE", ["breathing"], [], self.OWNER).urgency_tier == "EMERGENCY"

    def test_absent_with_real_quote_lifts_flag(self):
        checks = [check("breathing", "absent", "not struggling to breathe")]
        result = apply_floor("ROUTINE", ["breathing"], checks, self.OWNER)
        assert result.urgency_tier == "ROUTINE"
        assert result.flags_lifted == ["breathing"]

    def test_absent_with_made_up_quote_does_not_lift(self):
        checks = [check("breathing", "absent", "breathing is totally normal")]
        assert apply_floor("ROUTINE", ["breathing"], checks, self.OWNER).urgency_tier == "EMERGENCY"

    def test_never_lowers_model_urgency(self):
        checks = [check("breathing", "absent", "not struggling to breathe")]
        result = apply_floor("EMERGENCY", ["breathing"], checks, self.OWNER)
        assert result.urgency_tier == "EMERGENCY"
        assert not result.floor_applied

    def test_one_kept_flag_is_enough(self):
        checks = [
            check("breathing", "absent", "not struggling to breathe"),
            check("collapse", "present", "collapsed"),
        ]
        owner = "He's not struggling to breathe but he collapsed"
        result = apply_floor("ROUTINE", ["breathing", "collapse"], checks, owner)
        assert result.urgency_tier == "EMERGENCY"
        assert result.flags_kept == ["collapse"]


class TestLiftAgreement:
    def test_all_samples_agree(self):
        samples = [[check("breathing", "absent", "a b")]] * 3
        assert lift_agreement(["breathing"], samples) == {"breathing": 1.0}

    def test_partial_agreement(self):
        samples = [
            [check("breathing", "absent", "a b")],
            [check("breathing", "present", "a b")],
        ]
        assert lift_agreement(["breathing"], samples) == {"breathing": 0.5}

    def test_sample_that_skips_flag_counts_as_disagreeing(self):
        samples = [[check("breathing", "absent", "a b")], []]
        assert lift_agreement(["breathing"], samples) == {"breathing": 0.5}

    def test_mark_unclear_only_touches_named_flags(self):
        checks = [check("breathing", "absent", "a b"), check("collapse", "absent", "c d")]
        updated = mark_unclear(checks, {"breathing"})
        assert [c.status for c in updated] == ["unclear", "absent"]
