import json
import os
from llm_sdk import Small_LLM_Model
from .data_loader import load_data
from .decoder import generate_json
from .models import FunctionsCalling, FunctionsDef
from .parsing import parsing_arg


def main() -> None:
    """
    Function to run the full function-calling pipeline: parse the
    command-line arguments, load the LLM and its vocabulary, validate
    the input files, generate a JSON function call for every prompt
    using constrained decoding, and write the results to the output
    file.
    """
    args = parsing_arg()

    print("Loading LLM model...")
    llm = Small_LLM_Model()

    vocab_path = llm.get_path_to_vocab_file()
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab: dict[str, int] = json.load(f)

    functions = load_data(args.functions_definition, FunctionsDef)
    function_data = [func.model_dump() for func in functions]
    valid_func = [func["name"] for func in function_data]

    prompts = load_data(args.input, FunctionsCalling)
    prompt_data = [prompt.model_dump() for prompt in prompts]

    result = []

    for item in prompt_data:
        user_query = item["prompt"]
        system_prompt = (
            f"Available functions:\n{json.dumps(function_data)}\n\n"
            f"User Query: {json.dumps(user_query)}\n"
            f"Generate the strict JSON response:\n"
        )

        generate_json_str = generate_json(llm, system_prompt, vocab,
                                          valid_func, user_query,
                                          function_data)

        try:
            parsed_json = json.loads(generate_json_str.strip())
            result.append(parsed_json)
        except json.JSONDecodeError:
            result.append({
                "prompt": user_query,
                "name": "error_or_invalid_generation",
                "parameters": {}
            })
    output_dir = os.path.dirname(args.output) or "."
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
