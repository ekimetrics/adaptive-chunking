import langdetect

from adaptive_chunking.chunking_utils import is_high_confidence_non_english


class _Lang:
    def __init__(self, lang, prob):
        self.lang = lang
        self.prob = prob


def _patch_detect(monkeypatch, result):
    """Patch langdetect.detect_langs to return `result` or raise if it's an exception."""
    def fake(_text):
        if isinstance(result, Exception):
            raise result
        return result
    monkeypatch.setattr(langdetect, "detect_langs", fake)


class TestIsHighConfidenceNonEnglish:
    def test_high_confidence_non_english_is_true(self, monkeypatch):
        _patch_detect(monkeypatch, [_Lang("fr", 0.99)])
        assert is_high_confidence_non_english("du texte en français") is True

    def test_high_confidence_english_is_false(self, monkeypatch):
        _patch_detect(monkeypatch, [_Lang("en", 0.99)])
        assert is_high_confidence_non_english("some english text") is False

    def test_low_confidence_non_english_is_false(self, monkeypatch):
        # below threshold -> keep the document
        _patch_detect(monkeypatch, [_Lang("fr", 0.50)])
        assert is_high_confidence_non_english("ambiguous text") is False

    def test_empty_detection_is_false(self, monkeypatch):
        # Regression: previously read confidence/lang_code that were never
        # assigned when detection returned empty -> NameError / stale leak.
        _patch_detect(monkeypatch, [])
        assert is_high_confidence_non_english("") is False

    def test_detection_exception_is_false(self, monkeypatch):
        _patch_detect(monkeypatch, langdetect.LangDetectException(0, "no features"))
        assert is_high_confidence_non_english("???") is False

    def test_custom_threshold(self, monkeypatch):
        _patch_detect(monkeypatch, [_Lang("de", 0.90)])
        assert is_high_confidence_non_english("text", threshold=0.85) is True
        assert is_high_confidence_non_english("text", threshold=0.95) is False
