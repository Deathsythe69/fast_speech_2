"""Unit tests for Telugu text frontend (Phase 3)."""

import json

import pytest

from Telugu.text.g2p import phonemes_to_sequence_te, text_to_phonemes_te
from Telugu.text.normalizer import handle_zwnj_zwj, normalize_text_te
from Telugu.text.symbols import symbols

_valid_symbol_set = set(symbols)


@pytest.fixture
def test_sentences():
    with open("test_sentences.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["Telugu"]


def test_telugu_numeral_expansion(test_sentences):
    """Assert numeral expansion for '2026', '10:45', and '12వ' from T4."""
    t4 = test_sentences["T4"]
    norm_t4, _ = normalize_text_te(t4)

    # 2026 -> రెండు వేల ఇరవై ఆరు
    assert "రెండు వేల ఇరవై ఆరు" in norm_t4
    # 10:45 -> పది నలభై ఐదు
    assert "పది నలభై ఐదు" in norm_t4
    # 12వ -> పన్నెండవ
    assert "పన్నెండవ" in norm_t4


def test_zwnj_zwj_handling():
    """Assert ZWJ and ZWNJ are explicitly detected, handled, and logged."""
    sample_text = "రామ్\u200cకు మరియు లక్ష్మణ్\u200dకు"
    cleaned, logs = handle_zwnj_zwj(sample_text)

    # Both characters should be removed from cleaned text
    assert "\u200c" not in cleaned
    assert "\u200d" not in cleaned
    # Logs should capture ZWNJ and ZWJ counts
    assert len(logs) == 2
    log_chars = {item["char"] for item in logs}
    assert "ZWNJ" in log_chars
    assert "ZWJ" in log_chars


def test_all_five_telugu_sentences_no_unknown_symbols(test_sentences):
    """Assert all 5 Telugu test sentences produce valid phone sequences with no unknown symbols."""
    for key, text in test_sentences.items():
        phones = text_to_phonemes_te(text)
        assert len(phones) > 0, f"{key} produced empty phonemes"

        for p in phones:
            at_p = "@" + p
            assert (at_p in _valid_symbol_set) or (p in _valid_symbol_set), (
                f"Unknown symbol '{p}' produced for sentence {key}"
            )

        seq = phonemes_to_sequence_te(phones)
        assert len(seq) == len(phones)
        assert all(isinstance(idx, int) and idx >= 0 for idx in seq)


def test_homograph_pattu_handling(test_sentences):
    """Document and assert handling of homograph 'పట్టు' across T2a (grip) and T2b (silk).
    In Telugu orthography, both senses are segmental homographs (/p a t_ t_ u/).
    """
    t2a = test_sentences["T2a"]
    t2b = test_sentences["T2b"]

    phones_t2a = text_to_phonemes_te(t2a)
    phones_t2b = text_to_phonemes_te(t2b)

    expected_pattu = ["p", "a", "t_", "t_", "u"]

    def contains_subsequence(lst, sub):
        sub_len = len(sub)
        for i in range(len(lst) - sub_len + 1):
            if lst[i : i + sub_len] == sub:
                return True
        return False

    assert contains_subsequence(phones_t2a, expected_pattu)
    assert contains_subsequence(phones_t2b, expected_pattu)

