"""Telugu text normalizer for FastSpeech 2 frontend.
Handles Unicode NFC normalization, explicit ZWJ/ZWNJ handling,
and Telugu number/time/date/ordinal expansion.
"""

import re
import unicodedata

# Telugu digits and number words
TELUGU_NUMBERS = {
    0: "సున్నా",
    1: "ఒకటి",
    2: "రెండు",
    3: "మూడు",
    4: "నాలుగు",
    5: "ఐదు",
    6: "ఆరు",
    7: "ఏడు",
    8: "ఎనిమిది",
    9: "తొమ్మిది",
    10: "పది",
    11: "పదకొండు",
    12: "పన్నెండు",
    13: "పదమూడు",
    14: "పద్నాలుగు",
    15: "పదిహేను",
    16: "పదహారు",
    17: "పదిహేడు",
    18: "పద్దెనిమిది",
    19: "పంతొమ్మిది",
}

TELUGU_TENS = {
    20: "ఇరవై",
    30: "ముప్పై",
    40: "నలభై",
    50: "యాభై",
    60: "అరవై",
    70: "డెబ్బై",
    80: "ఎనభై",
    90: "తొంభై",
}


def number_to_telugu(n: int) -> str:
    """Recursively converts an integer to Telugu words."""
    if n in TELUGU_NUMBERS:
        return TELUGU_NUMBERS[n]
    if 20 <= n < 100:
        ten = (n // 10) * 10
        rem = n % 10
        if rem == 0:
            return TELUGU_TENS[ten]
        return f"{TELUGU_TENS[ten]} {TELUGU_NUMBERS[rem]}"
    if 100 <= n < 1000:
        h = n // 100
        rem = n % 100
        h_str = "వంద" if h == 1 else f"{TELUGU_NUMBERS[h]} వందల"
        if rem == 0:
            return h_str
        return f"{h_str} {number_to_telugu(rem)}"
    if 1000 <= n < 100000:
        th = n // 1000
        rem = n % 1000
        th_str = "వెయ్యి" if th == 1 else f"{number_to_telugu(th)} వేల"
        if rem == 0:
            return th_str
        return f"{th_str} {number_to_telugu(rem)}"
    if 100000 <= n < 10000000:
        lakh = n // 100000
        rem = n % 100000
        l_str = "లక్ష" if lakh == 1 else f"{number_to_telugu(lakh)} లక్షల"
        if rem == 0:
            return l_str
        return f"{l_str} {number_to_telugu(rem)}"
    return str(n)


def ordinal_to_telugu(n: int) -> str:
    """Converts a cardinal integer into its Telugu ordinal word (e.g. 12 -> పన్నెండవ)."""
    card = number_to_telugu(n)
    if card.endswith("ి") or card.endswith("ు"):
        return card[:-1] + "వ"
    elif card.endswith("ై") or card.endswith("ున్న"):
        return card + "వ"
    return card + "వ"


_te_time_re = re.compile(r"\b(\d{1,2}):(\d{2})\b")
_te_ordinal_re = re.compile(r"\b(\d+)వ\b")
_te_number_re = re.compile(r"\b\d+\b")
_whitespace_re = re.compile(r"\s+")


def expand_time_te(match) -> str:
    """Expands time e.g. 10:45 -> పది నలభై ఐదు."""
    h = int(match.group(1))
    m = int(match.group(2))
    h_str = number_to_telugu(h)
    m_str = number_to_telugu(m)
    return f"{h_str} {m_str}"


def expand_ordinal_te(match) -> str:
    """Expands ordinals e.g. 12వ -> పన్నెండవ."""
    n = int(match.group(1))
    return ordinal_to_telugu(n)


def expand_number_te(match) -> str:
    """Expands numbers e.g. 2026 -> రెండు వేల ఇరవై ఆరు."""
    n = int(match.group(0))
    return number_to_telugu(n)


def handle_zwnj_zwj(text: str) -> tuple[str, list[dict]]:
    """Handles Zero-Width Non-Joiner (\u200c) and Zero-Width Joiner (\u200d).
    Logs any removed or handled characters.
    """
    removed_logs = []

    # Count occurrences
    zwnj_count = text.count("\u200c")
    zwj_count = text.count("\u200d")

    if zwnj_count > 0:
        removed_logs.append({"char": "ZWNJ", "codepoint": "U+200C", "count": zwnj_count})
    if zwj_count > 0:
        removed_logs.append({"char": "ZWJ", "codepoint": "U+200D", "count": zwj_count})

    # In Telugu text, ZWNJ following virama (్\u200c) represents an explicit halant/chillus.
    # We replace '్\u200c' with standard virama '్'.
    cleaned = text.replace("్\u200c", "్").replace("్\u200d", "్")
    # Remove any stray ZWJ/ZWNJ
    cleaned = cleaned.replace("\u200c", "").replace("\u200d", "")

    return cleaned, removed_logs


def normalize_text_te(text: str) -> tuple[str, list[dict]]:
    """Full Telugu text normalizer pipeline:
    1. Unicode NFC normalization.
    2. ZWJ / ZWNJ explicit handling and logging.
    3. Number, time, and ordinal expansion.
    4. Whitespace cleanup.
    """
    # 1. Unicode NFC
    text = unicodedata.normalize("NFC", text)

    # 2. ZWJ / ZWNJ
    text, logs = handle_zwnj_zwj(text)

    # 3. Time, ordinal, and number expansion
    text = _te_time_re.sub(expand_time_te, text)
    text = _te_ordinal_re.sub(expand_ordinal_te, text)
    text = _te_number_re.sub(expand_number_te, text)

    # 4. Whitespace
    text = re.sub(_whitespace_re, " ", text).strip()

    return text, logs

