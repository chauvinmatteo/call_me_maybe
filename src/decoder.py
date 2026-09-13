import json
from llm_sdk import Small_LLM_Model
from .utils import get_ids_prefix, get_ids_free_mode, get_allowed_name_tokens


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
                  vocab: dict[str, int], valid_names: list[str]):

    with open("data/input/functions_definition.json") as f:
        functions_data = json.load(f)

    current_text = prompt
    current_state = 0
    generated_keys = []
    current_key = ""
    prompt_token = 0
    brace_depth = 0
    chosen_function_name = ""
    dynamic_param_keys = []

    while current_state < 7:
        prompt_id = llm.encode(current_text).tolist()[0]
        logit_list = llm.get_logits_from_input_ids(prompt_id)
        mask_list = [-float('inf')] * len(logit_list)

        if current_state == 0:
            allowed_ids = [vocab['{']]
        elif current_state == 1:
            allowed_ids = [vocab['"']]
        elif current_state == 2:
            current_word = current_text.split('"')[-1]
            allowed_keys = [k for k in ["prompt", "name", "parameters"] if k not in generated_keys]
            allowed_ids = get_allowed_name_tokens(allowed_keys, vocab, current_word)
            if current_word in allowed_keys:
                allowed_ids.append(vocab['"'])
                current_key = current_word
                if current_key not in generated_keys:
                    generated_keys.append(current_key)
        elif current_state == 3:
            allowed_ids = [vocab[':']]
        elif current_state == 4:
            if current_key == "parameters":
                allowed_ids = [vocab['{']]
            else:
                allowed_ids = [vocab['"']]
        elif current_state == 5:
            if current_key == "name":
                current_word = current_text.split('"')[-1]
                allowed_ids = get_allowed_name_tokens(valid_names, vocab, current_word)
                if current_word in valid_names:
                    allowed_ids.append(vocab['"'])
                    chosen_function_name = current_word
                    target_func = next((f for f in functions_data if f["name"] == chosen_function_name), None)
                    if target_func and "parameters" in target_func:
                        dynamic_param_keys = list(target_func["parameters"].keys())
            elif current_key == "prompt":
                prompt_token += 1
                allowed_ids = get_ids_free_mode(vocab)
                allowed_ids.append(vocab['"'])
                if prompt_token == 25:
                    allowed_ids = [vocab['"']]
            elif current_key == "parameters":
                current_word = current_text.split('"')[-1]
                allowed_ids = get_allowed_name_tokens(dynamic_param_keys, vocab, current_word)
                allowed_ids.append(vocab['"'])
                allowed_ids.append(vocab['}'])
                allowed_ids.append(vocab[':'])
                allowed_ids.append(vocab[','])
                allowed_ids.extend(get_ids_free_mode(vocab))
        elif current_state == 6:
            if len(generated_keys) < 3:
                allowed_ids = [vocab[',']]
            else:
                allowed_ids = [vocab['}']]

        for current_id in allowed_ids:
            mask_list[current_id] = logit_list[current_id]
        
        mask_max = max(mask_list)
        chosen_id = mask_list.index(mask_max)
        new_char = llm.decode([chosen_id])
        current_text += new_char
        print(new_char, end="", flush=True)

        if '{' in new_char:
            brace_depth += 1
        elif '}' in new_char:
            brace_depth -= 1

        if '"' in new_char and current_state in [1, 2, 4, 5]:
            current_state += 1
        elif '{' in new_char and current_state in [0, 4]:
            current_state += 1
        elif ':' in new_char and current_state == 3:
            current_state += 1
        elif ',' in new_char and current_state == 6:
            current_state = 1        
        elif '}' in new_char:
            if current_state == 5 and brace_depth == 0:
                current_state += 1
            elif current_state == 6:
                current_state = 7