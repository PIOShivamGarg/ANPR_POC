"""
Shared utility functions for ANPR processing.
"""
import re


def extract_state(text: str, ALL_STATE_NAMES: dict[str, str]) -> str | None:
    """
    Match any word or multi-word phrase in the OCR text
    against the global state names list (all countries).
    Returns properly cased state name (e.g. 'Washington', 'Texas') or None.
    
    Args:
        text: Raw OCR text from license plate
        ALL_STATE_NAMES: Dictionary mapping uppercase state names to proper case
        
    Returns:
        Properly cased state name or None if no match found
    """
    text_upper = text.upper()

    # First try multi-word matches (e.g. "New York", "New South Wales")
    for state_upper, state_proper in ALL_STATE_NAMES.items():
        if state_upper in text_upper:
            return state_proper

    # Fallback: single word match with punctuation stripped
    words = text_upper.split()
    for word in words:
        clean_word = re.sub(r'[^A-Z\s]', '', word).strip()
        if clean_word in ALL_STATE_NAMES:
            return ALL_STATE_NAMES[clean_word]

    return None


def extract_plate_number(text: str) -> str | None:
    """
    Extract license plate number from raw OCR text using Regex.
    Handles formats like:
      - CEH4091       (no separator)
      - FNR*8034      (star separator - Texas style)
      - ABC-1234      (hyphen separator)
      - ABC·1234      (dot/bullet separator)
      
    Args:
        text: Raw OCR text from license plate
        
    Returns:
        Cleaned plate number (without separators) or None if no valid plate found
    """
    pattern = r'\b([A-Z0-9]{2,4}[+*\-·]?[A-Z0-9]{2,4})\b'
    text_upper = text.upper()

    matches = re.findall(pattern, text_upper)

    if not matches:
        return None

    # Filter out false positives:
    # - Must have both letters and digits
    # - Minimum length 5 (without separator)
    # - Skip known noise words
    noise_words = {"READ", "TEXT", "ALPR", "STATE", "AUTO"}

    for match in matches:
        clean = re.sub(r'[+*\-·]', '', match)  # strip separators
        has_letter = bool(re.search(r'[A-Z]', clean))
        has_digit = bool(re.search(r'[0-9]', clean))
        is_noise = match in noise_words

        if has_letter and has_digit and not is_noise and len(clean) >= 5:
            return clean  # return cleaned plate number without separators

    return None


def create_plate_response(plate_number: str | None, state: str | None, 
                          region: str | None = None) -> dict:
    """
    Create standardized plate response dictionary.
    
    Args:
        plate_number: Extracted plate number
        state: Extracted state/region name
        region: Region detected by ALPR model (optional)
        
    Returns:
        Dictionary with plate_number, state, region, and message fields
    """
    message = None
    if not plate_number:
        message = "No plate number could be extracted from OCR text"
    
    return {
        "plate_number": plate_number,
        "state": state,
        "region": region,
        "message": message
    }
