import re
import unicodedata


def strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


def normalize(text: str) -> str:
    return strip_accents(text.lower())


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", normalize(text))
