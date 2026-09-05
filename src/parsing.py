import argparse


def parsing_arg() -> argparse.Namespace:
    """
    This function is used to parsed the different command-line arguments.

    Returns:
        argparse.Namespace: which are paths to desired files.
        One is for the functions_definitions, the second one for the input,
        and the last one for the output.
    """
    arg_parse = argparse.ArgumentParser()

    arg_parse.add_argument("--functions_definition",
                           default="data/input/functions_definition.json")
    arg_parse.add_argument("--input",
                           default="data/input/function_calling_tests.json")
    arg_parse.add_argument("--output",
                           default="data/output/function_calling_results.json")
    parsed_arg = arg_parse.parse_args()

    return parsed_arg
