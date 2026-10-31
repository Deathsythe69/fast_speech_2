"""Text processing module for Telugu FastSpeech 2."""

import re
from .symbols import symbols
from .normalizer import normalize_text_te
from .g2p import text_to_phonemes_te, phonemes_to_sequence_te

_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

_curly_re = re.compile(r"(.*?)\{(.+?)\}(.*)")


def text_to_sequence(text, cleaner_names=None):
    """Converts a string of Telugu text to a sequence of symbol IDs.
    Supports phoneme sequences enclosed in curly braces {p1 p2 p3}.
    If curly braces are not present, runs Telugu G2P.
    """
    sequence = []

    while len(text):
        m = _curly_re.match(text)
        if not m:
            phones = text_to_phonemes_te(text)
            sequence += phonemes_to_sequence_te(phones)
            break
        if m.group(1).strip():
            phones = text_to_phonemes_te(m.group(1))
            sequence += phonemes_to_sequence_te(phones)
        curly_phones = m.group(2).strip().split()
        sequence += phonemes_to_sequence_te(curly_phones)
        text = m.group(3)

    return sequence


def sequence_to_text(sequence):
    """Converts sequence of IDs back to phone string."""
    result = []
    for symbol_id in sequence:
        if symbol_id in _id_to_symbol:
            s = _id_to_symbol[symbol_id]
            if s.startswith("@"):
                s = s[1:]
            result.append(s)
    return " ".join(result)
