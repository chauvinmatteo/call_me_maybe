import json
from llm_sdk import Small_LLM_Model
from .utils import get_ids_prefix, get_ids_free_mode


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


def generate_json(llm: Small_LLM_Model, prompt: str,
                  vocab: dict[str, int]):

    current_text = prompt
    current_state = 0
    while current_state < 7:

        prompt_id = llm.encode(current_text).tolist()[0]
        logit_list = llm.get_logits_from_input_ids(prompt_id)
        mask_list = [-float('inf')] * len(logit_list)
        if current_state == 0:
            allowed_ids = get_ids_prefix("{", vocab)
        elif current_state == 1:
            allowed_ids = get_ids_prefix('"', vocab)
        elif current_state == 2:
            allowed_ids = get_ids_prefix('"', vocab) + get_ids_free_mode(vocab)
        elif current_state == 3:
            allowed_ids = get_ids_prefix(":", vocab)
        elif current_state == 4:
            allowed_ids = get_ids_prefix('"', vocab)
        elif current_state == 5:
            allowed_ids = get_ids_prefix('"', vocab) + get_ids_free_mode(vocab)
        else:
            allowed_ids = get_ids_prefix("}", vocab)

        for current_id in allowed_ids:
            mask_list[current_id] = logit_list[current_id]
        mask_max = max(mask_list)
        chosen_id = mask_list.index(mask_max)
        new_char = llm.decode([chosen_id])
        current_text += new_char
        print(new_char, end="", flush=True)
        if current_state == 0 or current_state == 1:
            current_state += 1
        elif current_state == 2 and '"' in new_char:
            current_state += 1
        elif current_state == 3 or current_state == 4:
            current_state += 1
        elif current_state == 5 and '"' in new_char:
            current_state += 1
        elif '}' in new_char:
            break
