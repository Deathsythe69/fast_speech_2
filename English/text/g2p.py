"""English G2P frontend for FastSpeech 2.
Integrates CMUdict, g2p_en with POS-based heteronym handling, and neural OOV fallback.
"""

from g2p_en import G2p

from .symbols import symbols

_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

# Initialize global G2p instance
_g2p_instance = None


def get_g2p():
    global _g2p_instance
    if _g2p_instance is None:
        _g2p_instance = G2p()
        # Enforce exact requirement: 'conduct' must be noun /K AA1 N D AH0 K T/ and verb /K AH0 N D AH1 K T/
        _g2p_instance.homograph2features["conduct"] = (
            ["K", "AH0", "N", "D", "AH1", "K", "T"],  # Verb
            ["K", "AA1", "N", "D", "AH0", "K", "T"],  # Noun
            "V",
        )
    return _g2p_instance


def text_to_phonemes_en(text):
    """Converts raw or normalized English text into a list of ARPAbet phoneme tokens.
    Handles POS-based heteronyms, neural OOV fallback, apostrophes, and punctuation.
    """
    g2p = get_g2p()
    # Normalize text first using normalizer
    from .normalizer import normalize_text_en
    norm_text = normalize_text_en(text)

    # Run g2p_en
    phones = g2p(norm_text)

    # Map spaces and punctuations into valid silence/punctuation tokens
    result = []
    for p in phones:
        if p == " " or p in [".", ",", "!", "?", ";", ":"]:
            result.append("sp")
        else:
            result.append(p)

    # Remove consecutive 'sp'
    clean_phones = []
    prev = None
    for p in result:
        if p == "sp" and prev == "sp":
            continue
        clean_phones.append(p)
        prev = p

    # Strip leading and trailing silences
    while clean_phones and clean_phones[0] == "sp":
        clean_phones.pop(0)
    while clean_phones and clean_phones[-1] == "sp":
        clean_phones.pop()

    return clean_phones


def phonemes_to_sequence_en(phonemes):
    """Converts a list of phonemes into symbol IDs.
    Prepends '@' to ARPAbet symbols and silences to match symbols.py.
    """
    sequence = []
    for p in phonemes:
        at_p = "@" + p
        if at_p in _symbol_to_id:
            sequence.append(_symbol_to_id[at_p])
        elif p in _symbol_to_id:
            sequence.append(_symbol_to_id[p])
        else:
            # Fallback to spoken noise
            sequence.append(_symbol_to_id.get("@spn", 0))
    return sequence

