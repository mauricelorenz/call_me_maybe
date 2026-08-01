*This project has been created as part of the 42 curriculum by mlorenz.*

# Call Me Maybe

## Description

This project implements a function-calling pipeline that translates natural language prompts into structured function calls using a small language model (Qwen/Qwen3-0.6B, 0.6B parameters). Given a question like "What is the sum of 2 and 3?", the system identifies the correct function (`fn_add_numbers`) and extracts the arguments (`{"a": 2.0, "b": 3.0}`), outputting a JSON object with the prompt, function name, and parameters.

The key challenge is guaranteeing 100% valid JSON output from a small model. This is achieved through **constrained decoding**: at each token generation step, the logits for all tokens that would violate the expected JSON schema are masked to negative infinity, forcing the model to only choose from structurally valid tokens.

## Instructions

### Prerequisites

- Python 3.10 or later
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

```bash
make install
# or equivalently:
uv sync
```

This installs all dependencies (numpy, pydantic, flake8, mypy, and the bundled llm_sdk).

### Running

```bash
make run
# or equivalently:
uv run python -m src
```

With optional arguments:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

### Debug

```bash
make debug
```

Runs the program under Python's built-in pdb debugger.

### Linting

```bash
make lint        # flake8 + mypy with standard flags
make lint-strict # flake8 + mypy --strict
```

### Clean

```bash
make clean
```

Removes `__pycache__`, `.mypy_cache`, and `.pyc` files.

## Resources

- [Qwen3-0.6B Model Card](https://huggingface.co/Qwen/Qwen3-0.6B) - The base model used for inference.
- [Pydantic Documentation](https://docs.pydantic.dev/) - Used for input/output validation.
- [Constrained Decoding / Outlines](https://blog.dlm.ai/2023/09/19/constrained-decoding.html) - Conceptual background on logit masking for structured output.
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/) - Model loading and tokenization.
- [NumPy](https://numpy.org/doc/) - Array operations for logit masking.
- [LLM Function Calling Concepts](https://platform.openai.com/docs/guides/function-calling) - Overview of function calling in LLMs.

**AI usage:** AI assistants were used for iterative development of the constrained decoding logic, debugging token masking edge cases, adding docstrings, and drafting this README. The core algorithm design and integration were done manually.

## Algorithm Explanation

The system uses a two-phase constrained decoding approach:

### Phase 1: Function Selection

1. A prompt is constructed listing all available function definitions.
2. The prompt is tokenized and fed to the LLM to obtain logits for the first token.
3. At each generation step, only tokens that could form a valid prefix of one of the known function names are allowed. All other tokens are masked to `-np.inf`.
4. The token with the highest remaining logit is selected.
5. Generation stops when the selected token sequence exactly matches one of the function name token sequences.

This guarantees the output is always exactly one of the defined function names.

### Phase 2: Parameter Extraction

1. A second prompt is constructed with the selected function's schema and the original user question.
2. The output JSON structure (`{"prompt": ..., "name": ..., "parameters": {`) is pre-generated as tokens and used as the initial context.
3. For each parameter in the function's schema:
   - The parameter key tokens (e.g., `"a":`) are injected directly.
   - The value is generated token-by-token with type-specific constraints:
     - **numbers**: Only tokens maintaining a valid numeric prefix (digits, decimal point, exponent notation) are allowed. A stop guard prevents incomplete numbers like `"5."` or `"5e"`.
     - **strings**: Only safe ASCII tokens are allowed. After a backslash token, only valid JSON escape characters (`"`, `\`, `/`, `b`, `f`, `n`, `r`, `t`) are permitted. Generation stops at a quote token.
     - **booleans**: Only the tokens `"true"` and `"false"` are allowed.
     - **integers**: Same as numbers but restricted to digit-only tokens.
4. Commas and closing braces are injected between and after parameters.

### Vocabulary Mapping

The tokenizer's vocabulary JSON is parsed at startup to build:
- An inverse mapping (`token_id -> string`) for reconstructing text from tokens.
- Categorized token lists (numbers, strings, booleans, quotes, escapes, etc.) used to build the masks at each step.

This ensures every generated token is structurally valid, producing 100% parseable JSON without relying on the model to spontaneously produce correct syntax.

## Design Decisions

1. **Two-phase generation over single-pass**: Function selection and parameter extraction are separated into distinct generation phases. This allows independent prompt engineering and type-specific constraint logic for each phase, rather than trying to enforce the entire JSON schema in one pass.

2. **Regex-based prefix validation for numbers**: Instead of maintaining a state machine, the system checks whether each candidate token would keep the current string a valid numeric prefix using precompiled regexes. This is simple, correct, and fast.

3. **Pydantic for input validation**: All input file parsing uses Pydantic models, providing structured validation with clear error messages rather than manual dictionary checks.

4. **Incremental file writing**: Results are written to the output file after each prompt is processed, so partial results are always preserved.

5. **Bundled llm_sdk as workspace member**: The SDK is included as a uv workspace member to ensure reproducible dependency management without requiring external package installation.

6. **Token-level masking over prompt engineering**: Rather than hoping the model produces valid JSON from instructions, the system enforces correctness at the logit level. This always achieves reliable output.

## Performance Analysis

- **Accuracy**: The constrained decoding approach achieves high reliability for function selection (always matches an exact function name) and parameter extraction (values are type-constrained). The small Qwen3-0.6B model handles the structured output well when guided by logit masking.
- **Speed**: Each prompt requires one forward pass per generated token. With the 0.6B model on CPU, typical processing takes a few seconds per prompt. All 11 test prompts complete well within the 5-minute budget.
- **JSON validity**: Every output is guaranteed to be structurally valid JSON conforming to the function schema, since invalid tokens are masked at every step. This is a hard guarantee, not a probabilistic one.

## Challenges Faced

- **Token boundary alignment**: Tokenizer tokens don't always align with logical JSON boundaries (a single token may include leading spaces, quotes, or multiple characters). The vocabulary mapping and per-type candidate filtering were designed to handle multi-character tokens correctly.
- **Number generation stopping condition**: Ensuring the model can stop generating a number at the right point (not mid-decimal like `"5."`) required a `stop_allowed` guard that checks whether the current string is a complete number before enabling the closing brace/comma tokens.
- **Escape sequence handling in strings**: JSON strings may contain escaped characters. The masking logic switches from "safe string" tokens to "escape" tokens after a backslash, and back to safe tokens after a valid escape character.
- **Vocabulary heterogeneity**: Different tokenizers represent the same characters differently (e.g., leading spaces encoded as `Ġ` in the raw vocab). The vocab loading step normalizes these representations for consistent mask construction.

## Testing Strategy

1. **Input validation**: Unit tests verify that `parse_infile` correctly validates well-formed JSON and rejects malformed input, missing files, and schema violations.
2. **Vocabulary mapping**: Tests confirm that token categorization correctly identifies number tokens, string-safe tokens, boolean tokens, and escape tokens from the vocabulary.
3. **End-to-end inference**: The provided test prompts in `data/input/function_calling_tests.json` are run through the full pipeline, and the output JSON is validated for structural correctness and schema compliance.
4. **Linting**: `make lint` runs flake8 and mypy with strict typing flags to catch type errors and style issues.

## Example Usage

```bash
# Run with default input/output paths
make run

# Run with custom paths
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

Output (written to `data/output/function_calling_results.json`):

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  },
  {
    "prompt": "Greet shrek",
    "name": "fn_greet",
    "parameters": {"name": "shrek"}
  },
  {
    "prompt": "Reverse the string 'hello'",
    "name": "fn_reverse_string",
    "parameters": {"s": "hello"}
  }
]
```
