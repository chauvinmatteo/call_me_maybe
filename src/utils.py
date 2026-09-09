
def get_ids_prefix(prefix: str, vocab: dict[str, int]) -> list[int]:

    valid_ids = []
    for text, token_id in vocab.items():
        if text.startswith(prefix):
            valid_ids.append(token_id)
    return valid_ids
