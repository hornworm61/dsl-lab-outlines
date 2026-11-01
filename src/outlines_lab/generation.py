"""Real CFG masking with scripted logits or an optional local LLM."""

import hashlib
import json
import math
import sys
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import llguidance
import numpy as np
from outlines.backends.llguidance import LLGuidanceLogitsProcessor

from .agent import AgentAttempt, AgentLoopError, AgentPrompt, run_agent_loop
from .language import grammar, parse
from .validation import Diagnostic

_MAX_PREVIOUS_PROPOSAL_CHARS = 1000
_EOS_STOP_PROBABILITY = 0.75
_EOS_PREFERENCE_START_FRACTION = 0.125
_FORCE_EOS_REMAINING_TOKENS = 64
_TEMPERATURE = 1.0
_TOP_K = 0
_TOP_P = 1.0


class _EosPreference:
    """Bias toward ending only when the grammar accepts EOS."""

    def __init__(self, processor, eos_token_id: int, max_new_tokens: int, torch) -> None:
        self.processor = processor
        self.eos_token_id = eos_token_id
        self.max_new_tokens = max_new_tokens
        self.torch = torch
        self.log_odds = math.log(_EOS_STOP_PROBABILITY / (1 - _EOS_STOP_PROBABILITY))
        self.preference_starts_after = int(max_new_tokens * _EOS_PREFERENCE_START_FRACTION)
        self.prompt_length: int | None = None

    def reset(self) -> None:
        self.processor.reset()
        self.prompt_length = None

    def __call__(self, input_ids, logits):
        masked_logits = self.processor(input_ids, logits)
        if self.prompt_length is None:
            self.prompt_length = input_ids.shape[-1]

        generated_tokens = input_ids.shape[-1] - self.prompt_length
        remaining_tokens = self.max_new_tokens - generated_tokens
        for row in range(masked_logits.shape[0]):
            eos_logit = masked_logits[row, self.eos_token_id]
            if not self.torch.isfinite(eos_logit):
                continue
            if remaining_tokens <= _FORCE_EOS_REMAINING_TOKENS:
                masked_logits[row].fill_(-self.torch.inf)
                masked_logits[row, self.eos_token_id] = 0
                continue
            if generated_tokens < self.preference_starts_after:
                continue

            other_logits = self.torch.cat(
                (
                    masked_logits[row, : self.eos_token_id],
                    masked_logits[row, self.eos_token_id + 1 :],
                )
            )
            other_logits = other_logits[self.torch.isfinite(other_logits)]
            if other_logits.numel():
                masked_logits[row, self.eos_token_id] = (
                    self.torch.logsumexp(other_logits, dim=0) + self.log_odds
                )
        return masked_logits


class GenerationError(ValueError):
    """Generation failed or ended before a complete DSL document."""


class _Characters:
    """Small inspectable tokenizer for the scripted-logit demonstration."""

    # Tabs let replay tests prove that the grammar rejects tab indentation.
    characters = "\n\r\t" + "".join(chr(i) for i in range(32, 127))
    tokens = [c.encode("ascii") for c in characters] + [b"<eos>"]
    eos_token_id = len(characters)
    bos_token_id = None
    special_token_ids = (eos_token_id,)

    def __call__(self, text: bytes | str) -> list[int]:
        if isinstance(text, bytes):
            text = text.decode("ascii")
        return [self.characters.index(c) for c in text]


@dataclass(frozen=True)
class MaskTrace:
    text: str
    blocked_tokens: int
    token_count: int


def _character_processor() -> tuple[_Characters, LLGuidanceLogitsProcessor]:
    characters = _Characters()
    tokenizer = llguidance.LLTokenizer(llguidance.TokenizerWrapper(characters))
    processor = LLGuidanceLogitsProcessor(
        llguidance.grammar_from("lark", grammar()), tokenizer, "numpy"
    )
    return characters, processor


def masked_replay(candidate: str) -> MaskTrace:
    """Offer an illegal CR above each scripted token; mask it using Outlines.

    This exercises the real mask, not a language model or a quality benchmark.
    No parser repairs or retries occur.
    """
    characters, processor = _character_processor()
    try:
        proposed = characters(candidate)
    except ValueError as exc:
        raise GenerationError(
            "Replay supports ASCII printable characters, tabs, and newlines only"
        ) from exc
    output: list[int] = []
    blocked = 0
    illegal = characters.characters.index("\r")
    for token in [*proposed, characters.eos_token_id]:
        logits = np.full((1, len(characters.tokens)), -np.inf, dtype=np.float32)
        logits[0, illegal] = 10
        logits[0, token] = 5
        masked = processor(np.array([output], dtype=np.int64), logits)
        if not np.isneginf(masked[0, illegal]):
            raise GenerationError("The deliberately illegal CR token was not masked")
        blocked += 1
        if not np.isfinite(masked[0, token]):
            raise GenerationError(f"Scripted token rejected at offset {len(output)}")
        selected = int(np.argmax(masked[0]))
        if selected == characters.eos_token_id:
            break
        output.append(selected)
    text = "".join(characters.characters[i] for i in output)
    parse(text)
    return MaskTrace(text, blocked, len(output) + 1)


REQUEST = (
    "Make a six-point two-fold dilution series of Sample S1 in wells A1-A6 of a "
    "96-well plate (200 uL capacity), add Reagent B, incubate for 15 minutes "
    "at 37 C, and measure absorbance at 450 nm. This is an invented software "
    "example, not a real experiment."
)


def _safetensors_hashes(model_path: Path) -> list[dict[str, str]]:
    weights = sorted(model_path.rglob("*.safetensors"))
    if not weights:
        raise GenerationError(f"No safetensors weights found in model directory: {model_path}")

    hashes = []
    for path in weights:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes.append({"file": str(path.relative_to(model_path)), "sha256": digest.hexdigest()})
    return hashes


def generate_local(
    model_path: Path,
    *,
    request: str = REQUEST,
    expected_fold: int | None = None,
    constrained: bool = True,
    max_new_tokens: int = 1024,
    max_attempts: int = 3,
    seed: int = 0,
    output_path: Path | None = None,
    transcript_path: Path | None = None,
    model_revision: str | None = None,
    stream_output: bool = False,
) -> str:
    if max_new_tokens <= 0:
        raise GenerationError("max_new_tokens must be positive")
    if max_attempts <= 0:
        raise GenerationError("max_attempts must be positive")
    if expected_fold is not None and expected_fold < 2:
        raise GenerationError("Expected dilution fold must be at least 2")
    if (
        output_path is not None
        and transcript_path is not None
        and output_path.resolve() == transcript_path.resolve()
    ):
        raise GenerationError("Output and transcript paths must be different")
    if output_path is not None and output_path.exists():
        raise GenerationError(f"Output path already exists: {output_path}")
    if transcript_path is not None and transcript_path.exists():
        raise GenerationError(f"Transcript path already exists: {transcript_path}")
    if not model_path.is_dir():
        raise GenerationError("Model must be an existing local directory; no automatic downloads")
    try:
        import outlines
        import torch
        from outlines.types import CFG
        from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, set_seed
    except ImportError as exc:
        raise GenerationError("Local generation requires: uv run --locked --extra local") from exc

    weights_sha256 = _safetensors_hashes(model_path) if transcript_path is not None else []
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False
    )
    if tokenizer.eos_token_id is None:
        raise GenerationError("Local generation requires a tokenizer with an EOS token")
    hf_model = AutoModelForCausalLM.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False, use_safetensors=True
    )
    model = outlines.from_transformers(hf_model, tokenizer)

    transcript_records: list[dict[str, str | int]] = []

    def propose(context: AgentPrompt) -> str:
        prompt = (
            "Return only a complete DSL document, without markdown or explanation. "
            "Follow the grammar exactly, including two-space command indentation "
            "and a final newline. The final serial transfer is retained in A6.\n\n"
            f"Grammar:\n{grammar()}\n\n"
            f"Original request:\n{context.request}\n"
        )
        if context.expected_fold is not None:
            prompt += (
                "\nEncoded semantic requirement: every serial-transfer step must "
                f"have a local dilution fold of {context.expected_fold}. Compute it "
                "as (receiving well volume before transfer + transfer volume) / "
                "transfer volume. For example, (50 + 50) / 50 = 2.\n"
            )
        if context.diagnostics:
            failure = (
                "grammar parsing"
                if any(item.code == "SYNTAX" for item in context.diagnostics)
                else "semantic validation"
            )
            diagnostic_counts: dict[str, int] = {}
            for item in context.diagnostics:
                diagnostic_counts[item.code] = diagnostic_counts.get(item.code, 0) + 1
            diagnostic_summary = ", ".join(
                f"{code} ({count})" if count > 1 else code
                for code, count in sorted(diagnostic_counts.items())
            )
            print(
                f"Retrying attempt {context.attempt}/{max_attempts}: the previous "
                f"proposal failed {failure} ({diagnostic_summary}). The previous "
                "proposal (up to 1,000 characters) and its diagnostics are included "
                "in the retry prompt.",
                file=sys.stderr,
                flush=True,
            )
            feedback = "\n".join(
                f"- {item.code} at {item.location}: {item.message}" for item in context.diagnostics
            )
            previous_proposal = context.previous_proposal or ""
            if len(previous_proposal) > _MAX_PREVIOUS_PROPOSAL_CHARS:
                previous_proposal = (
                    previous_proposal[:_MAX_PREVIOUS_PROPOSAL_CHARS] + "\n[... truncated ...]"
                )
            prompt += (
                f"\nYour previous proposal failed {failure}. Revise it to address "
                "these diagnostics while preserving the original request:\n"
                f"{previous_proposal}\n\nDiagnostics:\n{feedback}\n"
            )
        prompt += "\nProtocol:\n"
        if tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        inference_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": _TEMPERATURE,
            "top_k": _TOP_K,
            "top_p": _TOP_P,
        }
        if stream_output:

            class StderrTextStreamer(TextStreamer):
                def on_finalized_text(self, text: str, stream_end: bool = False) -> None:
                    if text:
                        sys.stderr.write(text)
                        sys.stderr.flush()
                    if stream_end:
                        sys.stderr.write("\n")
                        sys.stderr.flush()

            print(f"Attempt {context.attempt}: ", end="", file=sys.stderr, flush=True)
            inference_kwargs["streamer"] = StderrTextStreamer(
                tokenizer, skip_prompt=True, skip_special_tokens=True
            )
        text = generator(prompt, **inference_kwargs)
        if not isinstance(text, str):
            raise GenerationError(f"Expected one generated string, got {type(text).__name__}")
        transcript_records.append(
            {"attempt": context.attempt, "prompt": prompt, "raw_output": text}
        )
        return text

    def token_limit_diagnostic(proposal: str) -> Diagnostic | None:
        token_count = len(tokenizer.encode(proposal, add_special_tokens=False))
        if token_count >= max_new_tokens:
            return Diagnostic(
                "MAX_NEW_TOKENS",
                "generation",
                f"Output cut off at max_new_tokens={max_new_tokens}.",
            )
        return None

    generator = outlines.Generator(
        model, CFG(grammar()) if constrained else None, backend="llguidance"
    )
    eos_policy_installed = False
    logits_processor = getattr(generator, "logits_processor", None)
    if constrained:
        if logits_processor is None:
            raise GenerationError(
                "Constrained generation requires a logits processor to install the EOS policy"
            )
        eos_preference = _EosPreference(
            logits_processor,
            tokenizer.eos_token_id,
            max_new_tokens,
            torch,
        )
        generator.logits_processor = eos_preference
        if generator.logits_processor is not eos_preference:
            raise GenerationError("Could not install the constrained-generation EOS policy")
        eos_policy_installed = True

    def write_transcript(attempts: tuple[AgentAttempt, ...], *, outcome: str) -> None:
        if transcript_path is None:
            return
        if len(transcript_records) != len(attempts):
            raise GenerationError("Cannot write transcript: attempt records are incomplete")
        transcript_path.parent.mkdir(parents=True, exist_ok=True)
        with transcript_path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(
                json.dumps(
                    {
                        "record_type": "run",
                        "transcript_version": 1,
                        "outcome": outcome,
                        "model_path": str(model_path.resolve()),
                        "model_revision": model_revision,
                        "weights_sha256": weights_sha256,
                        "constrained": constrained,
                        "seed": seed,
                        "do_sample": True,
                        "temperature": _TEMPERATURE,
                        "top_k": _TOP_K,
                        "top_p": _TOP_P,
                        "eos_policy_installed": eos_policy_installed,
                        "eos_stop_probability": (
                            _EOS_STOP_PROBABILITY if eos_policy_installed else None
                        ),
                        "eos_preference_starts_after": (
                            int(max_new_tokens * _EOS_PREFERENCE_START_FRACTION)
                            if eos_policy_installed
                            else None
                        ),
                        "force_eos_remaining_tokens": (
                            _FORCE_EOS_REMAINING_TOKENS if eos_policy_installed else None
                        ),
                        "max_new_tokens": max_new_tokens,
                        "max_attempts": max_attempts,
                        "expected_fold": expected_fold,
                        "package_versions": {
                            "outlines-lab": version("outlines-lab"),
                            "outlines": version("outlines"),
                            "llguidance": version("llguidance"),
                            "transformers": version("transformers"),
                            "torch": version("torch"),
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            for record, attempt in zip(transcript_records, attempts, strict=True):
                status = (
                    "syntax_error"
                    if attempt.validation is None
                    else "passed"
                    if attempt.validation.valid
                    else "semantic_validation_failed"
                )
                stream.write(
                    json.dumps(
                        {
                            "record_type": "attempt",
                            **record,
                            "status": status,
                            "diagnostics": [
                                {
                                    "code": item.code,
                                    "location": item.location,
                                    "message": item.message,
                                }
                                for item in attempt.diagnostics
                            ],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    try:
        result = run_agent_loop(
            propose,
            request=request,
            expected_fold=expected_fold,
            max_attempts=max_attempts,
            truncation_diagnostic=token_limit_diagnostic,
        )
    except AgentLoopError as exc:
        write_transcript(exc.attempts, outcome="retry_limit_exhausted")
        mode = "Constrained" if constrained else "Unconstrained"
        raise GenerationError(
            f"{mode} agent loop failed: model output may contain malformed DSL, "
            f"prose, or Markdown, or may be truncated. No document saved.\n{exc}"
        ) from exc
    write_transcript(result.attempts, outcome="success")
    return result.proposal
