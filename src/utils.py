
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
