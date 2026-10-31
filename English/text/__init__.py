"""Text processing module for English FastSpeech 2."""

import re

from .g2p import phonemes_to_sequence_en, text_to_phonemes_en
from .normalizer import normalize_text_en
from .symbols import symbols

_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

_curly_re = re.compile(r"(.*?)\{(.+?)\}(.*)")


def text_to_sequence(text, cleaner_names=None):
    """Converts a string of text to a sequence of IDs corresponding to symbols.
    Supports ARPAbet phoneme sequences enclosed in curly braces {AA B C}.
    If curly braces are not present, uses the English G2P frontend.
    """
    sequence = []

    while len(text):
        m = _curly_re.match(text)
        if not m:
            # If plain text without curly braces, run through G2P
            phones = text_to_phonemes_en(text)
            sequence += phonemes_to_sequence_en(phones)
            break
        # Process prefix text before curly brace if any
        if m.group(1).strip():
            phones = text_to_phonemes_en(m.group(1))
            sequence += phonemes_to_sequence_en(phones)
        # Process ARPAbet enclosed in curly braces
        curly_phones = m.group(2).strip().split()
        sequence += phonemes_to_sequence_en(curly_phones)
        text = m.group(3)

    return sequence


def sequence_to_text(sequence):
    """Converts a sequence of IDs back to string."""
    result = []
    for symbol_id in sequence:
        if symbol_id in _id_to_symbol:
            s = _id_to_symbol[symbol_id]
            s = s.removeprefix("@")
            result.append(s)
    return " ".join(result)
