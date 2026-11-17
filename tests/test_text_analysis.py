import pytest

from src.audio.text_analysis import (LexiconSentiment, TransformersSentiment, analyze_text,
                                     evaluate_text, find_critical_terms, get_sentiment_analyzer,
                                     split_sentences)


def _cats(text, negated=False):
    return {h.category for h in find_critical_terms(text) if h.negated == negated}


def test_detects_basic_terms():
    assert _cats("I have chest pain and I feel dizzy.") >= {"chest_pain", "dizziness"}


def test_word_roots():
    assert "vomiting" in _cats("I vomited twice.")
    assert "fainting" in _cats("I fainted yesterday.")
    assert "fatigue" in _cats("I have been so tired lately.")


@pytest.mark.parametrize("text", [
    "No chest pain.", "I haven't had a fever.", "I don't feel short of breath.",
    "She denies any bleeding.", "No fever or chills.", "Never had headaches like this.",
    "Just a little nauseous, but no vomiting."])
def test_negated_terms(text):
    hs = find_critical_terms(text)
    assert hs and any(h.negated for h in hs)


def test_negation_does_not_leak_across_sentences():
    hs = find_critical_terms("No fever. I have chest pain.")
    chest = [h for h in hs if h.category == "chest_pain"][0]
    assert not chest.negated


def test_positive_symptom_with_distant_negation_is_kept():
    hs = [h for h in find_critical_terms("No, I really have been feeling a lot of chest pain lately") if h.category == "chest_pain"]
    assert hs and not hs[0].negated        # "no" fica a mais de 4 palavras


def test_cant_breathe_is_positive():
    assert "breathing_difficulty" in _cats("I can't breathe properly at night")


def test_case_insensitive_and_sentence_quoted():
    h = find_critical_terms("Honestly, CHEST PAIN started today. Then calm.")[0]
    assert h.category == "chest_pain" and h.sentence.startswith("Honestly")


def test_split_sentences():
    assert split_sentences("One. Two? Three!") == ["One.", "Two?", "Three!"]


def test_evaluate_text_findings_sorted_and_negated_ignored():
    a = analyze_text("I have severe pain in my arm. I feel a bit dizzy. No fever.", LexiconSentiment())
    f = evaluate_text(a)
    rules = [x.rule for x in f]
    assert "termo:severe_pain" in rules and "termo:dizziness" in rules and "termo:fever" not in rules
    assert f[0].severity == "high" and "severe pain" in f[0].message


def test_no_alert_when_everything_negated():
    a = analyze_text("No chest pain. I don't feel short of breath. Never fainted.", LexiconSentiment())
    assert [x for x in evaluate_text(a) if x.rule.startswith("termo:")] == []


def test_transformers_sentiment_with_injected_pipeline():
    class Pipe:
        def __call__(self, sents, **kw):
            return [{"label": "NEGATIVE", "score": 0.99} for _ in sents]
    an = TransformersSentiment(pipe=Pipe())
    a = analyze_text("It hurts. I feel awful. I am scared. Nothing helps. Very bad day.", an)
    assert a.negative_ratio == 1.0 and a.analyzer == "local"
    assert "tom_negativo" in [x.rule for x in evaluate_text(a)]


def test_sentiment_needs_min_segments():
    a = analyze_text("It hurts. I feel awful.", LexiconSentiment())
    assert "tom_negativo" not in [x.rule for x in evaluate_text(a)]


def test_factory(monkeypatch):
    monkeypatch.delenv("TEXT_ANALYZER", raising=False)
    assert get_sentiment_analyzer().name == "local"
    assert get_sentiment_analyzer("mock").name == "mock"
    monkeypatch.delenv("USE_AZURE", raising=False)
    with pytest.raises(RuntimeError, match="USE_AZURE"):
        get_sentiment_analyzer("azure")
    with pytest.raises(ValueError):
        get_sentiment_analyzer("xyz")
