"""Defines the set of symbols used in text input to the Telugu FastSpeech 2 model."""

_pad = "_"
_punctuation = "!'(),.:;? "
_special = "-"
_silences = ["@sp", "@spn", "@sil"]

# Telugu phoneme inventory
telugu_phonemes = [
    # Vowels
    "a",
    "aa",
    "i",
    "ii",
    "u",
    "uu",
    "ru",
    "ruu",
    "e",
    "ee",
    "ai",
    "o",
    "oo",
    "au",
    # Velars
    "k",
    "kh",
    "g",
    "gh",
    "ng",
    # Palatals
    "ch",
    "chh",
    "j",
    "jh",
    "ny",
    # Retroflexes
    "t_",
    "th_",
    "d_",
    "dh_",
    "n_",
    # Dentals
    "t",
    "th",
    "d",
    "dh",
    "n",
    # Labials
    "p",
    "ph",
    "b",
    "bh",
    "m",
    # Semivowels, liquids, fricatives
    "y",
    "r",
    "r_",
    "l",
    "l_",
    "v",
    "sh",
    "ss",
    "s",
    "h",
]

_telugu_phones = ["@" + p for p in telugu_phonemes]

# Export all symbols:
symbols = (
    [_pad]
    + list(_special)
    + list(_punctuation)
    + _telugu_phones
    + _silences
)
