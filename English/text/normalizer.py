"""English text normalizer for FastSpeech 2 frontend.
Handles number, date, time, abbreviation expansion while preserving apostrophes.
"""

import re

import inflect
from unidecode import unidecode

_inflect = inflect.engine()

_whitespace_re = re.compile(r"\s+")

# Abbreviations
_abbreviations = [
    (re.compile(r"\b%s\." % x[0], re.IGNORECASE), x[1])
    for x in [
        ("mrs", "missus"),
        ("mr", "mister"),
        ("dr", "doctor"),
        ("st", "saint"),
        ("co", "company"),
        ("jr", "junior"),
        ("maj", "major"),
        ("gen", "general"),
        ("drs", "doctors"),
        ("rev", "reverend"),
        ("lt", "lieutenant"),
        ("hon", "honorable"),
        ("sgt", "sergeant"),
        ("capt", "captain"),
        ("esq", "esquire"),
        ("ltd", "limited"),
        ("col", "colonel"),
        ("ft", "fort"),
        ("vs", "versus"),
        ("etc", "etcetera"),
        ("dept", "department"),
        ("univ", "university"),
    ]
]

MONTHS = r"(January|February|March|April|May|June|July|August|September|October|November|December)"
_date_re1 = re.compile(rf"\b(\d{{1,2}})(st|nd|rd|th)?\s+{MONTHS}\s+(\d{{4}})\b", re.IGNORECASE)
_date_re2 = re.compile(rf"\b{MONTHS}\s+(\d{{1,2}})(st|nd|rd|th)?\,?\s+(\d{{4}})\b", re.IGNORECASE)
_time_re = re.compile(r"\b(\d{1,2}):(\d{2})\s*(AM|PM|a\.m\.|p\.m\.|am|pm)?\b")

_comma_number_re = re.compile(r"([0-9][0-9\,]+[0-9])")
_decimal_number_re = re.compile(r"([0-9]+\.[0-9]+)")
_pounds_re = re.compile(r"£([0-9\,]*[0-9]+)")
_dollars_re = re.compile(r"\$([0-9\.\,]*[0-9]+)")
_ordinal_re = re.compile(r"\b([0-9]+)(st|nd|rd|th)\b")
_number_re = re.compile(r"\b[0-9]+\b")


def expand_year(num_str):
    num = int(num_str)
    if 1000 <= num <= 2999:
        if num % 1000 == 0:
            return _inflect.number_to_words(num)
        elif 2000 < num < 2010:
            return "two thousand " + _inflect.number_to_words(num % 100)
        elif num % 100 == 0:
            return _inflect.number_to_words(num // 100) + " hundred"
        else:
            return _inflect.number_to_words(num, andword="", zero="oh", group=2).replace(", ", " ")
    return _inflect.number_to_words(num)


def _expand_date1(m):
    day = int(m.group(1))
    month = m.group(3)
    year = m.group(4)
    day_word = _inflect.number_to_words(day)
    year_word = expand_year(year)
    return f"{day_word} {month} {year_word}"


def _expand_date2(m):
    month = m.group(1)
    day = int(m.group(2))
    year = m.group(4)
    day_word = _inflect.number_to_words(day)
    year_word = expand_year(year)
    return f"{month} {day_word} {year_word}"


def _expand_time(m):
    h = int(m.group(1))
    mins = int(m.group(2))
    meridiem = m.group(3)

    h_word = _inflect.number_to_words(h)
    if mins == 0:
        m_word = "o'clock" if not meridiem else ""
    elif mins < 10:
        m_word = "oh " + _inflect.number_to_words(mins)
    else:
        m_word = _inflect.number_to_words(mins)

    parts = [h_word]
    if m_word:
        parts.append(m_word)
    if meridiem:
        clean_meridiem = meridiem.replace(".", "").upper()
        parts.append(" ".join(list(clean_meridiem)))  # "A M" or "P M"
    return " ".join(parts)


def _remove_commas(m):
    return m.group(1).replace(",", "")


def _expand_decimal_point(m):
    return m.group(1).replace(".", " point ")


def _expand_dollars(m):
    match = m.group(1)
    parts = match.split(".")
    if len(parts) > 2:
        return match + " dollars"
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        cent_unit = "cent" if cents == 1 else "cents"
        return f"{dollars} {dollar_unit}, {cents} {cent_unit}"
    elif dollars:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        return f"{dollars} {dollar_unit}"
    elif cents:
        cent_unit = "cent" if cents == 1 else "cents"
        return f"{cents} {cent_unit}"
    else:
        return "zero dollars"


def _expand_ordinal(m):
    return _inflect.number_to_words(m.group(0))


def _expand_number(m):
    num = int(m.group(0))
    if 1000 < num < 3000:
        return expand_year(m.group(0))
    return _inflect.number_to_words(num, andword="")


def normalize_numbers(text):
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_pounds_re, r"\1 pounds", text)
    text = re.sub(_dollars_re, _expand_dollars, text)
    text = re.sub(_decimal_number_re, _expand_decimal_point, text)
    text = re.sub(_time_re, _expand_time, text)
    text = re.sub(_date_re1, _expand_date1, text)
    text = re.sub(_date_re2, _expand_date2, text)
    text = re.sub(_ordinal_re, _expand_ordinal, text)
    text = re.sub(_number_re, _expand_number, text)
    return text


def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text).strip()


def normalize_text_en(text):
    """Full English normalization pipeline:
    - Expand dates and times ('10:45 AM' -> 'ten forty-five A M', '12 November 2026' -> 'twelve November twenty twenty-six')
    - Expand abbreviations ('Mr.' -> 'mister')
    - Expand numbers
    - Preserve apostrophes ('orchestra\'s')
    - Collapse extra whitespace
    """
    # Normalize unicode characters safely
    text = unidecode(text)
    # Expand time and dates first before raw number matching
    text = re.sub(_time_re, _expand_time, text)
    text = re.sub(_date_re1, _expand_date1, text)
    text = re.sub(_date_re2, _expand_date2, text)
    text = expand_abbreviations(text)
    text = normalize_numbers(text)
    text = collapse_whitespace(text)
    return text

