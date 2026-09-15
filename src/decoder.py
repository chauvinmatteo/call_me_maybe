import json
from llm_sdk import Small_LLM_Model
from .utils import (get_ids_free_mode, get_ids_numeric_mode,
                    get_allowed_name_tokens)


def json_to_vocab_id(vocab_path: str) -> dict[str, int]:
    """
    Function to load a tokenizer vocabulary file into a token-text to
    token-id mapping.

    Args:
        vocab_path (str): Path to the vocabulary JSON file.

    Returns:
        dict[str, int]: Mapping of token text to token id.
    """
    with open(vocab_path) as f:
        vocab: dict[str, int] = json.load(f)
    return vocab


def generate_json(llm: Small_LLM_Model, system_prompt: str,
                  vocab: dict[str, int], valid_names: list[str],
                  user_query: str, functions_data: list[dict]) -> str:
    """
    Function to generate a single function-call JSON object for a user
    query, using constrained decoding so the result is always valid
    JSON that follows the function's schema.

    Args:
        llm (Small_LLM_Model): The language model wrapper used to get
        the next-token logits.
        system_prompt (str): The context given to the model, describing
        the available functions and the user query.
        vocab (dict[str, int]): Mapping of token text to token id.
        valid_names (list[str]): The exact function names the model is
        allowed to choose from.
        user_query (str): The original natural-language request.
        functions_data (list[dict]): The parsed functions_definition.json
        content, used to look up the parameters of the chosen function.

    Returns:
        str: The generated JSON object as text, without the
        system_prompt used to produce it.
    """

    free_ids = get_ids_free_mode(vocab)
    numeric_ids = get_ids_numeric_mode(vocab)
    integer_ids = get_ids_numeric_mode(vocab, allow_decimal=False)
    quote_id = vocab['"']

    current_text: str = system_prompt
    json_part_start = len(current_text)
    json_start = '{\n  "prompt": ' + json.dumps(user_query) + ',\n  "name": "'
    current_text += json_start

    current_ids = llm.encode(current_text).tolist()[0]

    chosen_function_name = ""
    while True:
        logit_list = llm.get_logits_from_input_ids(current_ids)
        mask_list = [-float('inf')] * len(logit_list)

        current_word = current_text.split('"')[-1]
        allowed_ids = get_allowed_name_tokens(valid_names, vocab, current_word)
        if current_word in valid_names:
            allowed_ids = [quote_id]

        for current_id in allowed_ids:
            mask_list[current_id] = logit_list[current_id]

        chosen_id = mask_list.index(max(mask_list))
        new_char = llm.decode([chosen_id])
        current_text += new_char
        current_ids.append(chosen_id)
        if current_word in valid_names and '"' in new_char:
            chosen_function_name = current_word
            break

    target_func = next((f for f in functions_data
                        if f["name"] == chosen_function_name), None)
    params_def = target_func.get("parameters", {}) if target_func else {}

    param_prefix = ',\n  "parameters": {'
    current_text += param_prefix

    for i, (param_name, param_info) in enumerate(params_def.items()):
        if i > 0:
            sep = ",\n  "
            current_text += sep

        key_part = f'"{param_name}": '
        current_text += key_part

        param_type = param_info.get("type", "number")

        if param_type == "string":
            current_text += '"'
            current_ids = llm.encode(current_text).tolist()[0]

            string_allowed_ids = free_ids + [quote_id]
            string_tokens = 0
            while True:
                logit_list = llm.get_logits_from_input_ids(current_ids)
                mask_list = [-float('inf')] * len(logit_list)

                for current_id in string_allowed_ids:
                    mask_list[current_id] = logit_list[current_id]

                chosen_id = mask_list.index(max(mask_list))
                new_char = llm.decode([chosen_id])
                string_tokens += 1

                if '"' in new_char or string_tokens >= 15:
                    if '"' not in new_char:
                        current_text += '"'
                        current_ids.append(quote_id)
                    else:
                        current_text += new_char
                        current_ids.append(chosen_id)
                    break

                current_text += new_char
                current_ids.append(chosen_id)
        elif param_type == "boolean":
            current_ids = llm.encode(current_text).tolist()[0]
            value_start = len(current_text)

            while True:
                logit_list = llm.get_logits_from_input_ids(current_ids)
                mask_list = [-float('inf')] * len(logit_list)

                current_word = current_text[value_start:]
                allowed_ids = get_allowed_name_tokens(
                    ["true", "false"], vocab, current_word)

                for current_id in allowed_ids:
                    mask_list[current_id] = logit_list[current_id]

                chosen_id = mask_list.index(max(mask_list))
                new_char = llm.decode([chosen_id])
                current_text += new_char
                current_ids.append(chosen_id)

                if current_text[value_start:] in ("true", "false"):
                    break
        else:
            is_integer = param_type == "integer"
            allowed_numeric_ids = integer_ids if is_integer else numeric_ids
            current_ids = llm.encode(current_text).tolist()[0]
            value_start = len(current_text)

            val_tokens = 0
            has_content = False
            while True:
                logit_list = llm.get_logits_from_input_ids(current_ids)
                mask_list = [-float('inf')] * len(logit_list)

                for current_id in allowed_numeric_ids:
                    mask_list[current_id] = logit_list[current_id]

                chosen_id = mask_list.index(max(mask_list))
                new_char = llm.decode([chosen_id])

                if has_content and any(c in new_char for c
                                       in [' ', '\n', ',', '}']):
                    break

                current_text += new_char
                current_ids.append(chosen_id)

                if new_char.strip():
                    has_content = True
                    val_tokens += 1

                if val_tokens >= 3:
                    break

            generated_value = current_text[value_start:]
            if not is_integer and not any(c in ".eE" for c in generated_value):
                current_text += ".0"

    suffix = '\n  }\n}'
    current_text += suffix

    return current_text[json_part_start:]
