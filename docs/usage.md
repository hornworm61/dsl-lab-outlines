# Repository usage guide

This guide covers the project environment, scripted generation simulation,
development checks, optional local-model generation, and auxiliary validation and
grammar tests.
Unless a command says otherwise, run it from the repository root.

Here, **DSL** means the domain-specific language. A **DSL document** is a file,
document, or text snippet that conforms to that language.

The project is an illustrative software demonstration. Its notation, reagents,
volumes, and limits are invented; it is not laboratory guidance, equipment safety
software, or executable robot control.

## CLI commands

| Command | Purpose |
| --- | --- |
| `outlines-lab generate --demo` | Run the scripted generation simulation; no LLM is loaded |
| `outlines-lab generate --model <dir> --output <file> [--stream] [--transcript <file>] [--model-revision REVISION]` | Generate and validate a proposal using a local model; optionally stream output and record a JSONL transcript |
| `outlines-lab validate <file> [--expected-fold N]` | Parse a DSL document and check encoded rules |

`generate --demo` runs the example-driven simulation described below. `generate`
without `--demo` uses the local-model options; see the generation setup in
section 4. Options such as `--request`, `--expected-fold`, and `--max-attempts`
apply to actual model generation, not to the fixed scripted simulation.

## 1. Set up the environment

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). Create the project
environment and install the locked runtime dependencies:

```sh
uv sync --locked
```

Installing dependencies or downloading model files may require internet access
if they are not already cached. After setup, the simulation runs locally with
scripted proposals; it does not download model weights, use a GPU, or contact an
inference service. The `validate` command also runs locally and needs no model.
`generate` without `--demo` needs the optional Transformers/PyTorch dependencies
and a compatible model already stored on your machine. It runs inference locally
without an external inference service, but the model's size determines the
memory and compute needed.

`uv run --locked --extra X` installs the extras it needs, so a separate `uv sync`
is optional.

## 2. Run the scripted generation simulation

```sh
uv run outlines-lab generate --demo
```

The simulation runs three proposal attempts: an over-capacity DSL document, a
within-capacity but wrong-fold DSL document, and one that passes the encoded checks.
The scripted responder chooses each revision from the previous validator
diagnostics; these are not LLM decisions. Each candidate DSL document is replayed through
Outlines' real LLGuidance logits processor with a tiny character tokenizer and
scripted scores. At each position, an illegal carriage-return token gets a higher
score than the candidate's next character. The mask blocks it and permits EOS
only when the document is complete. This demonstrates grammar masking and
diagnostic-driven retries, **not learned generation or generation quality**. The
blocked count is one CR proposal per character/EOS position, not all masked
vocabulary entries. The passing, unapproved proposal is handed off for human
intent review; the simulation does not perform that review.

| Candidate DSL document | Grammar / syntax | Encoded rules | Explicit two-fold check | Lesson |
| --- | --- | --- | --- | --- |
| [Malformed](../examples/00_malformed.lab) | Fail | Not run | Not run | Handwritten negative control, not generated output |
| [Original proposal](../examples/01_overflow.lab) | Pass | Fail | Fail | 210/220 uL exceeds the 200 uL limit |
| [Mechanical repair](../examples/02_wrong_intent.lab) | Pass | Pass | Fail | Ten-fold is well-formed but not two-fold |
| [Passing revision](../examples/03_two_fold.lab) | Pass | Pass | Pass | Encoded checks pass; human review remains |

After all steps, total liquid volume is 150 uL in A1-A5 and 200 uL in A6.

## 3. Development checks

These are for contributors changing code; they are not needed just to run the
simulation. Run tests, lint, and formatting checks with the development extra:

```sh
uv run --locked --extra dev pytest -q
uv run --locked --extra dev ruff check .
uv run --locked --extra dev ruff format --check .
```

The optional Transformers integration tests are skipped unless the `local` extra
is installed. They synthesize a tiny model locally and use scripted scores; they
test integration, not pretrained-model quality. Both constrained and unconstrained
local paths have been tested this way; no pretrained model evaluation is claimed.

## 4. Generate with an actual local model

### Install model dependencies

Install the optional Transformers and PyTorch dependencies in addition to the
runtime dependencies:

```sh
uv sync --locked --extra local
```

On Linux, the project configuration selects CPU PyTorch wheels. Model size
determines memory use and runtime; a larger model may require substantial RAM.
If you also want the development tools, install both extras with
`uv sync --locked --extra dev --extra local`.

### Prepare a model directory

Choose a license-compatible, instruction-tuned Hugging Face causal language model
supported by `AutoModelForCausalLM`. A tiny or random model can test integration,
but is unlikely to produce meaningful proposals. Download it using a trusted source,
or use a model snapshot already on your machine. The Hugging Face CLI is available
with the `local` extra; replace the repository ID and destination with your choices:

```sh
uv run --locked --extra local hf download organization/model-name \
  --local-dir /absolute/path/to/local-model
```

For gated models, authenticate with the provider's CLI before downloading; never
put access tokens in the repository or command history. The directory passed to
generation must contain the model configuration, tokenizer files, and
**safetensors** weights. Models requiring custom Python code are unsupported.
Ensure the model and tokenizer are compatible with the installed Transformers
version and that the model's context capacity can fit the request, grammar
instructions, and generated document.

The program loads only from the local directory using `local_files_only=True`,
disables remote code, and requires safetensors weights; it will not download a
model. It does not upload the model or prompt. Once the files are local, generation
does not require model-provider credentials.

#### Specific example: SmolLM2-360M-Instruct

With the `local` extra installed, download
[HuggingFaceTB/SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct)
at the pinned revision below into `~/models/SmolLM2-360M-Instruct`:

```sh
uv run --locked --extra local hf download HuggingFaceTB/SmolLM2-360M-Instruct \
  --revision a10cc1512eabd3dde888204e902eca88bddb4951 \
  --local-dir ~/models/SmolLM2-360M-Instruct
```

The model is instruction-tuned and distributed under Apache-2.0. The model files
require sufficient disk space. This command prepares a model directory. The
anecdotal run result is described below; it is not a model-quality evaluation.

### Generate a proposal

The SmolLM2 example below uses shell variables to make paired constrained and
unconstrained runs with the same model, revision, request, and sampling settings.
The constrained run additionally uses the EOS stop policy described below because
the grammar mask alone does not make the model terminate.

```sh
MODEL=~/models/SmolLM2-360M-Instruct
MODEL_REVISION=a10cc1512eabd3dde888204e902eca88bddb4951
REQUEST='Make a six-point two-fold dilution series of Sample S1 in wells A1-A6, add Reagent B, incubate for 15 minutes at 37 C, and measure absorbance at 450 nm. Keep each well at or below 200 uL.'

uv run --locked --extra local outlines-lab generate \
  --model "$MODEL" --model-revision "$MODEL_REVISION" --request "$REQUEST" \
  --expected-fold 2 --max-attempts 3 --max-new-tokens 1024 \
  --seed 0 --stream --output outputs/smollm2-constrained.lab \
  --transcript outputs/smollm2-constrained.jsonl

uv run --locked --extra local outlines-lab generate \
  --model "$MODEL" --model-revision "$MODEL_REVISION" --request "$REQUEST" \
  --expected-fold 2 --max-attempts 3 --max-new-tokens 1024 \
  --seed 0 --unconstrained --stream --output outputs/smollm2-unconstrained.lab \
  --transcript outputs/smollm2-unconstrained.jsonl
```

`--stream` writes each decoded text chunk to stderr as generation proceeds,
prefixed with the attempt number. It omits the prompt; complete raw responses are
still captured for validation and in the transcript. A streamed prefix may be
incomplete or invalid until generation ends.

These commands produce one seeded sampling run per mode. The grammar allows
commands and steps to continue without a fixed count, so grammar masking alone
does not make the model stop. In constrained mode, we also bias token selection
toward EOS (the end-of-sequence token): after one-eighth of the token budget, EOS
gets a 75% chance at grammar-accepted stopping points; with 64 or fewer tokens
remaining, EOS is forced at the first accepted stopping point. This encourages a
complete document to end rather than continuing indefinitely. The hard token
limit is still the ultimate bound; if reached before EOS is grammar-allowed,
generation fails rather than saving an incomplete document. The seed is recorded
for reproducibility; one run is anecdotal, not a rate or estimate of model
behavior.

A model this small may exhaust its retries; the transcript records why.

In the recorded seeded runs, SmolLM2-360M-Instruct exhausted all three attempts in
both modes, and neither run produced a semantically valid proposal. With grammar
masking, it produced syntactically valid DSL on each attempt, but the documents
failed encoded checks: examples included an 80,000 uL capacity, repeated step
names, ranges crossing plate rows, and reads from empty wells. Without the mask,
the attempts contained malformed DSL and explanatory prose and failed parsing.
This illustrates the key difference: grammar masking ensured syntactic form, but
did not make this small model satisfy the request or the semantic rules. These
are qualitative results from one seeded example, not a model-quality benchmark;
they do not establish that model size caused the failures.

The grammar mask is enabled by default. `--unconstrained` runs the same local
model without the mask for comparison. On each attempt, generation includes the
original request and any explicitly supplied expected-fold requirement. When
generation fails syntax or semantic validation, stderr explains why a retry is
starting. The next prompt includes the previous proposal and diagnostics; the
proposal is capped at 1,000 characters. Retries continue up to `--max-attempts`.

Both modes parse and validate each response. Syntax diagnostics include Lark's
line, column, and context and are fed back for another attempt. The command fails
without saving a proposal if the retry limit is reached without a valid result.
Without the mask (`--unconstrained`), the model may emit prose or Markdown, which
fails parsing. With the mask, a token budget can still cut off a valid prefix.
When an invalid output reaches the token budget, the retry diagnostic identifies
the cutoff; prior proposals are capped at 1,000 characters in the next prompt.

The optional JSONL transcript starts with a run record containing the outcome,
model path and declared revision, SHA-256 hashes of each safetensors weight file,
constrained mode, seed, sampling and EOS policy, token budget, retry limit,
expected fold, and versions of outlines-lab, Outlines, LLGuidance, Transformers,
and PyTorch. The `--model-revision`
value is recorded metadata; it does not select or verify the model files. Verify
the recorded weight hashes against the files to check that the same weights were
used. Each subsequent attempt record contains the attempt number, rendered prompt,
raw model output, status, and diagnostics.
The transcript is written on success and when the retry limit is exhausted, so
failed attempts are preserved. Failures before generation starts do not produce
a transcript. Prompts include the request; protect transcripts accordingly.
Output and transcript paths must not already exist. A passing proposal is still
unapproved: compare it with the original request and the
[human review checklist](review-checklist.md). The fold requirement is not
inferred from prose; specify `--expected-fold` when it is required.

## 5. Auxiliary validation and grammar checks

These commands are useful when inspecting examples or changing the grammar; they
are not required to run the simulation.

Validate examples with and without the explicit two-fold requirement:

```sh
uv run outlines-lab validate examples/02_wrong_intent.lab
uv run outlines-lab validate examples/02_wrong_intent.lab --expected-fold 2
uv run outlines-lab validate examples/03_two_fold.lab --expected-fold 2
```

Exit codes are `0` for passing encoded rules, `1` for an encoded-rule failure,
and `2` for syntax, input, or generation errors. A parser error stops semantic
validation. Passing checks do not prove the proposal matches the full intent.

Run grammar compatibility tests:

```sh
uv run --locked --extra dev --extra local pytest -q tests/test_grammar_compatibility.py
```

The [white paper's Appendix A](white-paper.md#appendix-a-parser-and-tokenizer-checks)
describes the grammar-compatibility corpus and checks. With the `local` extra
installed, a tokenizer-only test trains byte-level BPE locally and checks tokens
spanning grammar boundaries; no model weights are needed for that test. This is
not a pretrained-tokenizer evaluation.

## Project layout and scope

- [Shared grammar](../src/outlines_lab/protocol.lark): a deliberately small,
  line-oriented subset of the paper's notation.
- [Parser and domain model](../src/outlines_lab/language.py).
- [Volume and optional intent validator](../src/outlines_lab/validation.py).
- [Bounded generation/validation feedback loop](../src/outlines_lab/agent.py).
- [Outlines integration](../src/outlines_lab/generation.py).
- [Tests](../tests/): masking, syntax, exact volumes, transient overflow, intent,
  diagnostic feedback, bounded retries, and CLI behavior.
- [White paper](white-paper.md): architecture, evidence, qualifications, and references.

The default validator intentionally does **not** infer requirements from prose.
The optional fold check encodes only the local volume-based dilution factor at each
transfer, not all experimental intent. Reagent identity, step completeness/order,
point count, temperature, timing, measurement suitability, and the effect of
development on final concentrations still need review or richer formal requirements.

## License and attribution

Original project code and documentation contributions are licensed under the
[MIT License](../LICENSE), copyright (c) 2026 Steven Livengood. Third-party source
material retains its own terms; see [NOTICE](../NOTICE) for scope and attribution.
Dependencies and user-supplied model assets retain their respective licenses.

The worked example and DSL-mediated-agent pattern are adapted from Steve Livengood,
*DSL-mediated Agents: A Definition* (2026), [DOI](https://doi.org/10.5281/zenodo.23587494),
licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The
[author's source](https://github.com/thirdrailsoftware/dsl-mediated-agents/blob/main/definition.md)
provides the full definition and broader related work.
