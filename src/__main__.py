import argparse


def main() -> None:
    """Run the main program."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--functions_definition",
                        default="data/input/functions_definition.json",
                        help="path for the functions definition json file")
    parser.add_argument("--input",
                        default="data/input/function_calling_tests.json",
                        help="path for the input json file")
    parser.add_argument("--output",
                        default="data/output/function_calls.json",
                        help="path for the output json file")
    # args = parser.parse_args()


if __name__ == "__main__":
    main()
