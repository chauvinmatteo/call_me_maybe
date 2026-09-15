*This project has been created as part of the 42 curriculum by mchauvin.*

### Call Me Maybe

## Description

This project implements a function-calling system for small LLMs: given a
natural-language prompt and a list of available functions, it uses
constrained decoding on Qwen/Qwen3-0.6B to produce a JSON object naming the
right function and its arguments, guaranteed to be valid JSON. Instead of
asking the model to "just write JSON" and hoping, the code inspects the
model's logits at every generation step and forbids, in advance, any token
that would break the JSON structure or the function's schema.

## Instructions

### Installation

```sh
make install
```

### Running

```sh
make run
```

By default this reads `data/input/functions_definition.json` and
`data/input/function_calling_tests.json`, and writes
`data/output/function_calling_results.json`. Custom paths:

```sh
uv run python -m src --functions_definition <path> --input <path> --output <path>
```

### Other Makefile targets

- `make debug` — run the main script under `pdb`.
- `make lint` / `make lint-strict` — flake8 + mypy.
- `make clean` — remove `__pycache__` / `.mypy_cache`.

## Algorithm Explanation

The JSON object is never generated in one shot. `generate_json()` (in
[decoder.py](src/decoder.py)) builds it field by field: the fixed parts of
the structure (`{`, `"prompt": ...`, key names, commas, closing braces) are
written directly by the Python code, and the model is only asked to fill in
the two parts that actually depend on its judgment: the function name and
each parameter's value.

**Choosing the function name.** At every step, `get_allowed_name_tokens()`
(in [utils.py](src/utils.py)) looks at what has been typed so far and keeps
only the vocabulary tokens that are a valid continuation of at least one of
the real function names. Every other token's logit is forced to `-inf`
before the model picks its next token, so it is structurally impossible for
the model to "hallucinate" a function name that doesn't exist — the moment
it types a wrong character, there is nothing left to legally continue with.
Once the typed text matches a full name exactly, the only token still
allowed is the closing quote.

**Filling in a parameter value.** Once the name is known,
`functions_definition.json` tells us each parameter's declared type, and
`generate_json()` branches on it:

- `string` — any character is allowed except `"`, `\`, and raw control
  characters (`get_ids_free_mode`). Generation stops on a closing quote, or
  after 15 tokens as a safety net.
- `number` — only digits, `-` and `.` are allowed (`get_ids_numeric_mode`).
  If nothing generated contains a `.`, one is appended afterwards, so the
  value always parses back as a Python `float`.
- `integer` — same as `number`, but `.` is never in the allowed character
  set to begin with, so the value always parses back as an `int`.
- `boolean` — reuses the same prefix-matching logic as the function name,
  against the two-item list `["true", "false"]`.

**Escaping the prompt.** The user's original query is embedded into the
JSON with `json.dumps(user_query)` rather than an f-string, so a prompt
that itself contains a `"` (a real case in the test set) cannot break the
JSON structure — `json.dumps` escapes quotes, backslashes and control
characters automatically.

## Design Decisions

Constrained decoding was used instead of prompting-and-hoping because a
0.6B model asked to freely write JSON gets it right only a fraction of the
time — masking invalid tokens out of the logits makes invalid output
structurally impossible instead of merely unlikely, which is the actual
point of this exercise.

Each generation loop (string, number) has a maximum number of tokens as a
safety net, in case the model never naturally produces a stopping
character (a closing quote, a space, a comma). The number cap was raised
from 3 to 12 tokens after testing showed 3 was too aggressive and silently
truncated a legitimate large number (`1234567.89` became `123.0`) — the cap
exists to bound worst-case runaway generation, not to limit normal values.

The biggest performance lever available without modifying the `llm_sdk`
package (which is off-limits) was avoiding redundant work in the Python
code around the model calls: the token ids fed to the model are tracked
incrementally (`current_ids.append(chosen_id)`) instead of re-tokenizing
the whole accumulated text on every step, and the vocabulary scans that
decide which tokens are safe to use (`get_ids_free_mode`,
`get_ids_numeric_mode`) are computed once per run in `main()` instead of
once per prompt.

## Performance Analysis

**Speed.** A full run over 11 test prompts takes roughly 4 minutes on this
machine (no GPU available), comfortably under the 5-minute budget. The
dominant cost is outside this project's control: `get_logits_from_input_ids`
re-runs the model's entire forward pass on every single generated token,
with no KV-cache to reuse work between tokens — a change only possible
inside `llm_sdk` itself.

**Reliability.** Every generated object is syntactically valid,
schema-typed JSON: no `error_or_invalid_generation` fallback has appeared
in any test run, including prompts specifically containing nested quotes,
Windows-style paths, and large numbers.

**Accuracy.** Measured against a grading tool that calls the real target
function with the generated arguments and compares the output: 8/11
(72.7%) on the public function set, 6/11 (54.5%) on a private set with
extra `integer`/`boolean` types and trickier prompts. Every remaining
failure is the same pattern: on functions needing several string
parameters (e.g. writing a regex from a natural-language description), the
model writes a correct start and then keeps generating invented,
Python-call-shaped text instead of stopping. The JSON stays valid in every
case — only the *content* of that one value is wrong. This is a capability
ceiling of a 0.6B model doing free-form character generation, not a
decoding bug: nothing forces the model to know the difference between
`\d+` and English words describing "numbers".

## Challenges Faced

- **Unescaped quotes breaking the JSON.** The user prompt was originally
  interpolated into an f-string. A test prompt containing a literal `"`
  closed the JSON string early and broke every output containing one.
  Fixed by using `json.dumps(user_query)` instead.
- **Every single prompt failing to parse.** `generate_json()` used to
  return the entire accumulated text, including the whole system prompt
  (the full function list) in front of the actual JSON object —
  `json.loads` failed on character 0, on every prompt, for a reason that
  had nothing to do with the model. Fixed by remembering the index where
  the real JSON object starts and slicing from there.
- **A raw newline leaking into a generated string.** The tokenizer used by
  Qwen represents control characters (space, tab, newline) with special
  stand-in characters internally (e.g. `Ċ` for `\n`), not the literal
  character. A filter that checked for a literal `"\n"` in the raw
  vocabulary text never matched anything. Fixed by decoding each candidate
  token back to real text with `llm.decode()` and checking that text
  directly, which also sidesteps needing to know anything about how the
  tokenizer stores its vocabulary internally.
- **Numeric type mismatches.** A JSON number with no decimal point parses
  back in Python as an `int`, not a `float`. The grading functions assert
  `isinstance(x, float)`, so a numerically correct answer like `"a": 2`
  still failed. Fixed by appending `.0` when a `number`-typed value has no
  `.`, and by never allowing a `.` at all for `integer`-typed values.
- **Braces wrongly banned from strings.** An earlier version excluded `{`,
  `}` and `:` from string values on the assumption they were
  JSON-structural and dangerous. They're not — inside an already-quoted
  string, JSON does not care about brace balance. Excluding them meant the
  model could never write a template placeholder like `{user}`.
- **A `mypy` false positive.** `mypy .` resolves `llm_sdk` to the outer
  project directory (which has no `__init__.py`) instead of the real,
  properly-installed package one level deeper, because that directory
  looks like an implicit namespace package. Fixed by adding
  `--no-namespace-packages` to the Makefile's `MYPY_FLAGS`.

## Testing Strategy

The implementation was tested by running the full pipeline against
`data/input/` and manually inspecting every generated object for JSON
validity and correct types, then targeting specific edge cases as they
were suspected: prompts containing embedded double quotes, prompts with
Windows-style backslash paths, very large and very small numbers, and
functions with `integer`/`boolean` parameters. It was also validated
end-to-end against an external grading tool that parses the output,
re-calls the real target function with the generated arguments, and
compares the result to the expected one — not just checking that the JSON
parses, but that the call itself is correct.

## Example Usage

```sh
$ make run
Loading LLM model...
```

Example input (`data/input/function_calling_tests.json`):

```json
[
  { "prompt": "What is the sum of 2 and 3?" }
]
```

Example output (`data/output/function_calling_results.json`):

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": { "a": 2.0, "b": 3.0 }
  }
]
```

## Resources

- [Hugging Face `transformers` docs on `generate()` and logits
  processors](https://huggingface.co/docs/transformers/generation_strategies) —
  useful background on how constrained/guided generation is normally done,
  even though this project implements the masking by hand instead of using
  a `LogitsProcessor`.
- [RFC 8259 — The JSON Data Interchange
  Format](https://www.rfc-editor.org/rfc/rfc8259) — the exact grammar used
  to reason about which characters a JSON string may or may not contain
  unescaped.
- [Qwen3 model card on Hugging
  Face](https://huggingface.co/Qwen/Qwen3-0.6B).

### AI Usage

An AI assistant (Claude) was used throughout this project's debugging and
documentation phase, specifically for:

- Basic explanation on different context.
- Helping with normes error.
- Helping to make the program run faster and smoother.
- Writing this readme.

