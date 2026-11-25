"""
Shared utility functions for ANPR processing.
"""
import re

# Tokens that look plate-shaped but never are.
_NOISE_TOKENS = {"READ", "TEXT", "ALPR", "STATE", "AUTO", "USA", "DMV", "EXP"}

# Registration year sticker, e.g. "2025" / "1998".
_YEAR_RE = re.compile(r'^(19|20)\d{2}$')

# Unicode-aware word tokens (keeps accented state names like "FÈS" intact).
_TOKEN_RE = re.compile(r'[^\W_]+', re.UNICODE)

# Canonical plate shape: letters-then-digits or digits-then-letters.
_PLATE_SHAPE = re.compile(r'^([A-Z]{1,4}[0-9]{2,5}|[0-9]{1,4}[A-Z]{2,4})$')

_MAX_STATE_WORDS = 4

# A plate number is this many alphanumerics. The shortest genuine read across the
# sample set is 6; international layouts (MH12AB1234) run to 10.
_MIN_PLATE_LEN = 5
_MAX_PLATE_LEN = 10

# Below this, fast_alpr's region prediction is a guess: genuine plates in testing
# all reported >= 0.98, while a redacted Texas plate returned "Belgium" at 0.22.
_MIN_REGION_CONFIDENCE = 0.5


def extract_state(text: str, ALL_STATE_NAMES: dict[str, str]) -> str | None:
    """
    Match a whole word or multi-word phrase in the OCR text against the global
    state names list (all countries), preferring the longest phrase.
    Returns properly cased state name (e.g. 'Washington', 'Texas') or None.

    Matching is word-boundary based and longest-first, so a short name
    ('Van', 'Ra') can neither match inside plate characters nor beat the
    longer name that actually appears ('Ohio', 'New York').

    Args:
        text: Raw OCR text from license plate
        ALL_STATE_NAMES: Dictionary mapping uppercase state names to proper case

    Returns:
        Properly cased state name or None if no match found
    """
    words = _TOKEN_RE.findall(text.upper())

    for size in range(min(_MAX_STATE_WORDS, len(words)), 0, -1):
        for i in range(len(words) - size + 1):
            chunk = words[i:i + size]
            # Some names are hyphenated ("Al-Qahirah"); OCR gives them as words.
            for phrase in (" ".join(chunk), "-".join(chunk)):
                if phrase in ALL_STATE_NAMES:
                    return ALL_STATE_NAMES[phrase]

    return None


def _is_plausible_plate(clean: str | None) -> bool:
    """
    Could this string be a plate number at all? Length bounds plus a noise-word
    check — the minimum bar every candidate must clear, whichever OCR read it.
    """
    return bool(clean) and (_MIN_PLATE_LEN <= len(clean) <= _MAX_PLATE_LEN
                            and clean not in _NOISE_TOKENS)


def confident_region(region: str | None,
                     region_confidence: float | None) -> str | None:
    """
    The detector's region/country prediction, or None when it is only guessing.
    """
    if not region or (region_confidence or 0.0) < _MIN_REGION_CONFIDENCE:
        return None
    return region


def _plate_score(clean: str, parts: list[str]) -> int:
    """
    Score a candidate plate number. 0 means "not a plate".
    Higher is better; used to pick between competing OCR fragments.
    """
    if not _is_plausible_plate(clean):
        return 0
    if any(part in _NOISE_TOKENS or _YEAR_RE.match(part) for part in parts):
        return 0
    if not re.search(r'[A-Z]', clean) or not re.search(r'[0-9]', clean):
        return 0

    score = 1
    if _PLATE_SHAPE.match(clean):
        score += 2  # letters/digits cleanly grouped, as on a real plate
    if len(parts) == 1:
        score += 1  # OCR read it as one token — stronger evidence
    return score


def extract_plate_candidate(
    text: str, state: str | None = None
) -> tuple[str, list[str]] | None:
    """
    Extract a license plate number from raw OCR text, together with the OCR
    tokens it was built from (so the caller can look up their confidence).

    The OCR of a full-plate crop returns the plate mixed in with state name,
    slogan and year sticker, e.g.
      "WASHINGTON 4 :2025 CHF 4653 . EVERGREEN STATE ."
    so candidates are built from single tokens AND adjacent token pairs, then
    scored. Any non-alphanumeric separator (space, '*', '-', '·') is dropped.

    Handles formats like:
      - CEH4091       (no separator)
      - CHF 4653      (space separator - split across OCR lines)
      - FNR*8034      (star separator - Texas style)
      - ABC-1234      (hyphen separator)
      - ABC·1234      (dot/bullet separator)

    Args:
        text: Raw OCR text from license plate
        state: State name already matched in the text, if any; removed from the
               candidate pool so it cannot be mistaken for plate characters.

    Returns:
        (plate_number, source_tokens) or None if no valid plate found
    """
    text_upper = text.upper()
    if state:
        text_upper = re.sub(re.escape(state.upper()), " ", text_upper)

    tokens = re.findall(r'[A-Z0-9]+', text_upper)
    candidates = ([(t, [t]) for t in tokens]
                  + [(a + b, [a, b]) for a, b in zip(tokens, tokens[1:])])

    # max() keeps the leftmost of equally scored candidates.
    scored = [(_plate_score(clean, parts), clean, parts) for clean, parts in candidates]
    score, clean, parts = max(scored, key=lambda c: c[0], default=(0, "", []))
    return (clean, parts) if score else None


def extract_plate_number(text: str, state: str | None = None) -> str | None:
    """
    Extract license plate number from raw OCR text.
    Thin wrapper over extract_plate_candidate for callers that only want the
    number. See that function for the formats handled.

    Returns:
        Cleaned plate number (without separators) or None if no valid plate found
    """
    candidate = extract_plate_candidate(text, state)
    return candidate[0] if candidate else None


def _normalize_plate(text: str | None) -> str | None:
    """Uppercase, strip everything that is not a letter or digit."""
    return re.sub(r'[^A-Z0-9]', '', text.upper()) or None if text else None


def _min_confidence(confidence: float | list[float] | None) -> float | None:
    """
    Collapse a confidence to a single pessimistic number.
    fast_alpr reports one value per character; the weakest character is what
    decides whether the whole plate is trustworthy.
    """
    values = ([confidence] if isinstance(confidence, (int, float))
              else [float(c) for c in confidence or []])
    return round(min(values), 3) if values else None


def reconcile_plate_number(
    alpr_text: str | None,
    alpr_confidence: float | list[float] | None,
    azure_text: str | None,
    azure_confidence: float | None,
) -> dict:
    """
    Pick a plate number from two independent OCR sources.

    fast_alpr is a plate-specialised recogniser; Azure CV is general OCR that
    also reads the surrounding state/slogan text. They usually agree; when they
    do not, the more confident read wins. On an exact tie fast_alpr wins, since
    reading plates is the only thing it does. A source with text but no
    confidence is treated as unmeasured (0.0) and loses to any measured read.
    Either read is discarded unless it is plausibly a plate number at all; a
    blacked-out plate makes fast_alpr emit fragments of the state name.

    Returns:
        Dict with plate_number, confidence, source ('alpr' | 'azure' | 'both')
        and alternate_plate_number (the rejected read, only on disagreement).
    """
    def result(plate, confidence, source, alternate=None):
        return {
            "plate_number": plate,
            "confidence": confidence,
            "source": source,
            "alternate_plate_number": alternate,
        }

    # fast_alpr first, so an exact confidence tie falls to it (sorting is stable).
    reads = [(_normalize_plate(alpr_text), _min_confidence(alpr_confidence), "alpr"),
             (_normalize_plate(azure_text), _min_confidence(azure_confidence), "azure")]
    present = [r for r in reads if _is_plausible_plate(r[0])]

    if not present:
        return result(None, None, None)
    if len(present) == 1:
        return result(*present[0])

    if present[0][0] == present[1][0]:
        # Two independent engines agreeing corroborates the read.
        confidences = [c for _, c, _ in present if c is not None]
        return result(present[0][0], max(confidences) if confidences else None, "both")

    winner, loser = sorted(present, key=lambda r: r[1] or 0.0, reverse=True)
    return result(*winner, loser[0])


def create_plate_response(plate_number: str | None, state: str | None,
                          region: str | None = None,
                          confidence: float | None = None,
                          source: str | None = None,
                          alternate_plate_number: str | None = None,
                          message: str | None = None) -> dict:
    """
    Create standardized plate response dictionary.

    Args:
        plate_number: Extracted plate number
        state: Extracted state/region name
        region: Region detected by ALPR model (optional)
        confidence: Confidence of the winning OCR read (optional)
        source: Which OCR source won - 'alpr', 'azure' or 'both' (optional)
        alternate_plate_number: The rejected read, when the sources disagreed
        message: Overrides the default message (e.g. why detection failed)

    Returns:
        Dictionary with plate_number, state, region, confidence, source,
        alternate_plate_number and message fields
    """
    if message is None:
        if not plate_number:
            message = "No plate number could be extracted from OCR text"
        elif alternate_plate_number:
            message = (
                f"OCR sources disagreed; kept the more confident read "
                f"({source}={plate_number}, rejected {alternate_plate_number})"
            )

    return {
        "plate_number": plate_number,
        "state": state,
        "region": region,
        "confidence": confidence,
        "source": source,
        "alternate_plate_number": alternate_plate_number,
        "message": message
    }
