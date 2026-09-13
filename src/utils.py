
def get_ids_prefix(prefix: str, vocab: dict[str, int]) -> list[int]:

    valid_ids = []
    for text, token_id in vocab.items():
        if text.startswith(prefix):
            valid_ids.append(token_id)
    return valid_ids


def get_ids_free_mode(vocab: dict[str, int]):

    unwanted_ids = ["{", "}", ":", '"']
    valid_ids = []
    for text, token_id in vocab.items():
        if not any(char in text for char in unwanted_ids):
            valid_ids.append(token_id)
    return valid_ids

def get_allowed_name_tokens(valid_names: list[str], vocab: dict[str, int],
                            current_word: str):

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
