from src.ui.callout import callout_words


def test_recommended_plan_keeps_the_word():
    label, verb = callout_words(7, 7)
    assert label == "Recommended Plan (Solution #7):"
    assert verb == "the system recommends"


def test_inspected_plan_does_not_say_recommended():
    label, verb = callout_words(12, 7)
    assert "Recommended" not in label and "recommend" not in verb
    assert label == "Plan you are inspecting (Solution #12):"
    assert verb == "this plan makes"
