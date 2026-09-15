
_CONTROL_CHARS = {chr(b) for b in range(32)}
_CONTROL_SURROGATES = {chr(256 + b) for b in range(32)}
_CONTROL_LIKE = _CONTROL_CHARS | _CONTROL_SURROGATES


def get_ids_prefix(prefix: str, vocab: dict[str, int]) -> list[int]:
    """
    Function to collect every vocabulary token whose text starts with
    a given prefix.

    Args:
        prefix (str): The prefix a token's text must start with.
        vocab (dict[str, int]): Mapping of token text to token id.

    Returns:
        list[int]: Token ids of every matching token.
    """
    valid_ids = []
    for text, token_id in vocab.items():
        if text.startswith(prefix):
            valid_ids.append(token_id)
    return valid_ids


def get_ids_free_mode(vocab: dict[str, int]) -> list[int]:
    """
    Function to collect every vocabulary token that is safe to emit
    inside a JSON string value: no JSON structural character
    (``{``, ``}``, ``:``, ``"``), no backslash, and no raw control
    character (tab, carriage return, line feed, etc.).

    Args:
        vocab (dict[str, int]): Mapping of token text to token id.

    Returns:
        list[int]: Token ids that are safe to use inside a JSON string.
    """
    unwanted_chars = {"{", "}", ":", '"', "\\"}
    valid_ids = []
    for text, token_id in vocab.items():
        if any(char in unwanted_chars for char in text):
            continue
        if any(char in _CONTROL_LIKE for char in text):
            continue
        valid_ids.append(token_id)
    return valid_ids


def get_ids_numeric_mode(vocab: dict[str, int],
                         allow_decimal: bool = True) -> list[int]:
    """
    Function to collect every vocabulary token that is either a pure
    JSON-number character sequence or a pure delimiter (whitespace or
    comma), so a generated number value can never end up containing an
    invalid character or an unescaped control character.

    Args:
        vocab (dict[str, int]): Mapping of token text to token id.
        allow_decimal (bool): Whether the generated number may contain
        '.', 'e'/'E' or a sign character. True for a JSON "number"
        (float), False for a strict JSON "integer".

    Returns:
        list[int]: Token ids that are safe to use while generating a
        number value.
    """
    numeric_chars = (set("0123456789.eE+-") if allow_decimal
                     else set("0123456789-"))
    delimiter_chars = set(" \n\t,") | _CONTROL_SURROGATES
    valid_ids = []
    for text, token_id in vocab.items():
        if text and (all(c in numeric_chars for c in text)
                     or all(c in delimiter_chars for c in text)):
            valid_ids.append(token_id)
    return valid_ids


def get_allowed_name_tokens(valid_names: list[str], vocab: dict[str, int],
                            current_word: str) -> list[int]:
    """
    Function to collect every vocabulary token that can legally extend
    the partial function name generated so far, i.e. whose text is a
    prefix of the remaining part of at least one valid function name.

    Args:
        valid_names (list[str]): The exact names the model is allowed
        to choose from.
        vocab (dict[str, int]): Mapping of token text to token id.
        current_word (str): The partial function name generated so far.

    Returns:
        list[int]: Token ids that can validly continue current_word.
    """
    remaining_suffixes = []
    valid_ids = []
    for name in valid_names:
        if name.startswith(current_word):
            suffix = name[len(current_word):]
            remaining_suffixes.append(suffix)
    for text, token_id in vocab.items():
        for suffix in remaining_suffixes:
            if suffix.startswith(text):
                valid_ids.append(token_id)
                break
    return valid_ids
