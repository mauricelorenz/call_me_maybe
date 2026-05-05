import argparse
from src.parsing import parse_infile
from src.inference import generate_outfile

def main() -> None:
    """Run the main program."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--functions_definition",
                        default="data/input/functions_definition.json",
                        help="path for the functions definition JSON file")
    parser.add_argument("--input",
                        default="data/input/function_calling_tests.json",
                        help="path for the input JSON file")
    parser.add_argument("--output",
                        default="data/output/function_calls.json",
                        help="path for the output JSON file")
    args = parser.parse_args()
    functions_definition = parse_infile(args.functions_definition)
    input = parse_infile(args.input)
    output_path = args.output
    generate_outfile(functions_definition, input, output_path)


if __name__ == "__main__":
    main()
