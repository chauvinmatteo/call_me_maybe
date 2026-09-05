import json


def json_to_vocab_id(vocab_path: str) -> dict[str, int]:

    with open(vocab_path) as f:
        data = json.load(f)
        tokens_utils = {
            "{": data["{"],
            "}": data["}"],
            '"': data['"'],
            ":": data[":"],
            ",": data[","]
        }

    return tokens_utils
