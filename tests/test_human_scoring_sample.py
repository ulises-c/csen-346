"""human_scoring_sample: cleaning translator output."""

from scripts.human_scoring_sample import clean_translation


def test_plain_english_is_kept():
    assert (
        clean_translation("Do you think air can be weighed?") == "Do you think air can be weighed?"
    )


def test_echoed_chinese_and_label_are_removed():
    raw = "你觉得空气能不能被称重呢？\n\nTranslation: Do you think air can be weighed or not?"
    assert clean_translation(raw) == "Do you think air can be weighed or not?"


def test_output_with_no_english_after_the_chinese_is_rejected():
    assert clean_translation("你觉得空气能不能被称重呢？") is None
