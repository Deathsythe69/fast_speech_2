"""Telugu rule-based G2P (Grapheme-to-Phoneme) converter for FastSpeech 2.
Maps Telugu Unicode characters into deterministic phonetic phoneme sequences.
Correctly handles virama, consonant conjuncts, vowel matras, and anusvara.
"""

from .symbols import symbols

_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

# Vowel maps
INDEPENDENT_VOWELS = {
    "అ": "a",
    "ఆ": "aa",
    "ఇ": "i",
    "ఈ": "ii",
    "ఉ": "u",
    "ఊ": "uu",
    "ఋ": "ru",
    "ౠ": "ruu",
    "ఎ": "e",
    "ఏ": "ee",
    "ఐ": "ai",
    "ఒ": "o",
    "ఓ": "oo",
    "ఔ": "au",
}

DEPENDENT_MATRAS = {
    "ా": "aa",
    "ి": "i",
    "ీ": "ii",
    "ు": "u",
    "ూ": "uu",
    "ృ": "ru",
    "ౄ": "ruu",
    "ె": "e",
    "ే": "ee",
    "ై": "ai",
    "ొ": "o",
    "ో": "oo",
    "ౌ": "au",
}

CONSONANTS = {
    "క": "k",
    "ఖ": "kh",
    "గ": "g",
    "ఘ": "gh",
    "ఙ": "ng",
    "చ": "ch",
    "ఛ": "chh",
    "జ": "j",
    "ఝ": "jh",
    "ఞ": "ny",
    "ట": "t_",
    "ఠ": "th_",
    "డ": "d_",
    "ఢ": "dh_",
    "ణ": "n_",
    "త": "t",
    "థ": "th",
    "ద": "d",
    "ధ": "dh",
    "న": "n",
    "ప": "p",
    "ఫ": "ph",
    "బ": "b",
    "భ": "bh",
    "మ": "m",
    "య": "y",
    "ర": "r",
    "ఱ": "r_",
    "ల": "l",
    "ళ": "l_",
    "వ": "v",
    "శ": "sh",
    "ష": "ss",
    "స": "s",
    "హ": "h",
}

VIRAMA = "్"
ANUSVARA = "ం"
VISARGA = "ః"


def text_to_phonemes_te(text: str) -> list[str]:
    """Converts Telugu text into phonetic phoneme tokens."""
    from .normalizer import normalize_text_te

    # Normalize text first
    norm_text, _ = normalize_text_te(text)

    phonemes = []
    i = 0
    n = len(norm_text)

    while i < n:
        ch = norm_text[i]

        if ch in INDEPENDENT_VOWELS:
            phonemes.append(INDEPENDENT_VOWELS[ch])
            i += 1
        elif ch in CONSONANTS:
            base_phone = CONSONANTS[ch]
            # Check for following character
            if i + 1 < n and norm_text[i + 1] == VIRAMA:
                # Suppress inherent vowel (pure consonant / conjunct coda)
                phonemes.append(base_phone)
                i += 2
                # Skip any trailing ZWNJ/ZWJ
                if i < n and norm_text[i] in ["\u200c", "\u200d"]:
                    i += 1
            elif i + 1 < n and norm_text[i + 1] in DEPENDENT_MATRAS:
                # Replace inherent vowel with matra vowel
                phonemes.append(base_phone)
                phonemes.append(DEPENDENT_MATRAS[norm_text[i + 1]])
                i += 2
            else:
                # Inherent vowel 'a'
                phonemes.append(base_phone)
                phonemes.append("a")
                i += 1
        elif ch == ANUSVARA:
            phonemes.append("m")
            i += 1
        elif ch == VISARGA:
            phonemes.append("h")
            i += 1
        elif ch in " ,.!?-\"':;":
            phonemes.append("sp")
            i += 1
        elif ch in ["\u200c", "\u200d"]:
            i += 1
        else:
            # Skip unknown symbols gracefully
            i += 1

    # Clean consecutive and edge silences
    cleaned = []
    prev = None
    for p in phonemes:
        if p == "sp" and prev == "sp":
            continue
        cleaned.append(p)
        prev = p

    while cleaned and cleaned[0] == "sp":
        cleaned.pop(0)
    while cleaned and cleaned[-1] == "sp":
        cleaned.pop()

    return cleaned


def phonemes_to_sequence_te(phonemes: list[str]) -> list[int]:
    """Converts Telugu phonemes to sequence IDs using symbols.py."""
    seq = []
    for p in phonemes:
        at_p = "@" + p
        if at_p in _symbol_to_id:
            seq.append(_symbol_to_id[at_p])
        elif p in _symbol_to_id:
            seq.append(_symbol_to_id[p])
        else:
            # Fallback silence
            seq.append(_symbol_to_id.get("@sp", 0))
    return seq

