"""Small pure functions (no heavy libraries) so they are easy to test."""
import re
import unicodedata

MAX_REWRITES = 1  # how many times we rephrase the question before giving up

NOT_FOUND = {"ar": "ما لقيت هذي المعلومة في الملفات المرفوعة.",
             "en": "I couldn't find this in the uploaded files."}


def clean(text: str) -> str:
    # NFKC maps Arabic presentation forms (common in PDFs) back to normal letters
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\u200b-\u200f\u202a-\u202e]", "", text)  # invisible bidi marks
    return re.sub(r"[ \t]+", " ", re.sub(r"\n{3,}", "\n\n", text)).strip()


def detect_language(question: str) -> str:
    arabic = len(re.findall(r"[\u0600-\u06FF]", question))
    return "ar" if arabic > len(question) * 0.3 else "en"


def route_after_grade(state: dict) -> str:
    if state["relevant"]:
        return "answer"
    return "rewrite" if state["tries"] < MAX_REWRITES else "not_found"


def cite(meta: dict) -> str:
    """'p.341' or 'p.341 (printed 317)' when the PDF's printed page number differs from its position."""
    page, label = meta.get("page"), str(meta.get("label", ""))
    return f"p.{page}" if not label or label == str(page) else f"p.{page} (printed {label})"