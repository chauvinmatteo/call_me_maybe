import argparse


def parsing_arg() -> argparse.Namespace:

    arg_parse = argparse.ArgumentParser()

    arg_parse.add_argument("--functions_definition",
                           default="data/input/functions_definition.json")
    arg_parse.add_argument("--input",
                           default="data/input/function_calling_tests.json")
    arg_parse.add_argument("--output",
                           default="data/output/function_calling_results.json")
    parsed_arg = arg_parse.parse_args()

    return parsed_arg
