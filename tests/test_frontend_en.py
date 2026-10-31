"""Unit tests for English text frontend (Phase 3)."""

import json
import pytest
from English.text.normalizer import normalize_text_en
from English.text.g2p import text_to_phonemes_en, phonemes_to_sequence_en
from English.text.symbols import symbols

_valid_symbol_set = set(symbols)


@pytest.fixture
def test_sentences():
    with open("test_sentences.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["English"]


def test_normalization_dates_and_times(test_sentences):
    """Assert '10:45 AM' -> 'ten forty-five A M' and '12 November 2026' -> 'twelve November twenty twenty-six'."""
    e4 = test_sentences["E4"]
    norm_e4 = normalize_text_en(e4)

    assert "ten forty-five A M" in norm_e4 or "ten forty-five a m" in norm_e4.lower()
    assert "twelve November twenty twenty-six" in norm_e4 or "twelve november twenty twenty-six" in norm_e4.lower()
    assert "andaroo" in norm_e4


def test_apostrophe_preservation(test_sentences):
    """Assert apostrophe in orchestra's is preserved."""
    e2a = test_sentences["E2a"]
    norm_e2a = normalize_text_en(e2a)
    assert "orchestra's" in norm_e2a


def test_conduct_heteronym_distinction(test_sentences):
    """Assert conduct in E2a is noun /K AA1 N D AH0 K T/ and in E2b is verb /K AH0 N D AH1 K T/."""
    e2a = test_sentences["E2a"]
    e2b = test_sentences["E2b"]

    phones_e2a = text_to_phonemes_en(e2a)
    phones_e2b = text_to_phonemes_en(e2b)

    # Expected sub-sequences
    noun_conduct = ["K", "AA1", "N", "D", "AH0", "K", "T"]
    verb_conduct = ["K", "AH0", "N", "D", "AH1", "K", "T"]

    # Verify noun conduct in E2a
    def contains_subsequence(lst, sub):
        sub_len = len(sub)
        for i in range(len(lst) - sub_len + 1):
            if lst[i : i + sub_len] == sub:
                return True
        return False

    assert contains_subsequence(phones_e2a, noun_conduct), f"E2a phones {phones_e2a} did not contain noun /K AA1 N D AH0 K T/"
    assert contains_subsequence(phones_e2b, verb_conduct), f"E2b phones {phones_e2b} did not contain verb /K AH0 N D AH1 K T/"
    assert phones_e2a != phones_e2b


def test_all_five_english_sentences_no_unknown_symbols(test_sentences):
    """Assert all 5 English test sentences produce valid phone sequences with no unknown symbols."""
    for key, text in test_sentences.items():
        phones = text_to_phonemes_en(text)
        assert len(phones) > 0, f"{key} produced empty phonemes"
        
        # Check every phoneme is valid
        for p in phones:
            at_p = "@" + p
            assert (at_p in _valid_symbol_set) or (p in _valid_symbol_set), (
                f"Unknown symbol '{p}' produced for sentence {key}"
            )

        seq = phonemes_to_sequence_en(phones)
        assert len(seq) == len(phones)
        assert all(isinstance(idx, int) and idx >= 0 for idx in seq)


def test_oov_words_handling():
    """Assert OOV words ('antidisestablishmentarianism', 'andaroo', 'rhyth', 'm') do not crash."""
    oovs = ["antidisestablishmentarianism", "andaroo", "rhyth", "m"]
    for word in oovs:
        phones = text_to_phonemes_en(word)
        assert len(phones) > 0
        seq = phonemes_to_sequence_en(phones)
        assert len(seq) > 0

