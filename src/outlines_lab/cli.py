"""Simulate or run grammar-constrained generation with semantic validation."""

import argparse
import json
import sys
from pathlib import Path

from .agent import AgentLoopError, AgentPrompt, run_agent_loop
from .generation import REQUEST, GenerationError, generate_local, masked_replay
from .language import ProtocolSyntaxError, parse
from .validation import validate


def _read(path: Path) -> str:
    return path.read_bytes().decode("utf-8")


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def _fold(value: str) -> int:
    number = int(value)
    if number < 2:
        raise argparse.ArgumentTypeError("Dilution fold must be at least 2")
    return number


def _run_simulation(examples: Path) -> int:
    print("SIMULATION ONLY: scripted proposals, not an actual language model.")
    print("ILLUSTRATIVE ONLY: no robot instructions or real experimental guidance.")
    malformed = _read(examples / "00_malformed.lab")
    try:
        parse(malformed)
    except ProtocolSyntaxError:
        print("00_malformed: syntax FAIL (handwritten negative control, not LLM output)")
    else:
        raise GenerationError("Malformed negative control unexpectedly parsed")

    traces = []

    def propose(context: AgentPrompt) -> str:
        codes = {diagnostic.code for diagnostic in context.diagnostics}
        if not codes:
            name = "01_overflow"
        elif "CAPACITY" in codes:
            name = "02_wrong_intent"
        elif "INTENT_FOLD" in codes:
            name = "03_two_fold"
        else:
            raise GenerationError(f"No scripted response for diagnostics: {sorted(codes)}")
        trace = masked_replay(_read(examples / f"{name}.lab"))
        traces.append(trace)
        return trace.text

    result = run_agent_loop(
        propose,
        request=REQUEST,
        expected_fold=2,
        max_attempts=3,
    )

    for attempt_number, attempt in enumerate(result.attempts, 1):
        trace = traces[attempt_number - 1]
        validation = attempt.validation
        print(
            f"Attempt {attempt_number}: grammar "
            f"{'PASS' if validation is not None else 'FAIL'}; semantic validation "
            f"{'NOT RUN' if validation is None else 'PASS' if validation.valid else 'FAIL'}."
        )
        print(
            f"  The real Outlines mask blocked one CR proposal at each of "
            f"{trace.blocked_tokens} positions (including EOS). These are scripted "
            "logits, not an LLM."
        )
        if validation is not None:
            for diagnostic in validation.diagnostics:
                print(f"  {diagnostic.code} {diagnostic.location}: {diagnostic.message}")
        if (validation is None or not validation.valid) and attempt_number < len(result.attempts):
            print("  Feeding these diagnostics into the next generation prompt.")
    print(
        f"Semantic validation passed after {len(result.attempts)} attempts. "
        "Handing the unapproved proposal to a human for intent review."
    )
    print(f"  Final volumes: {result.validation.volumes}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("validate", help="Parse and check encoded rules")
    check.add_argument("protocol", type=Path)
    check.add_argument("--expected-fold", type=_fold)

    generate = commands.add_parser(
        "generate", help="Simulate or run generation with semantic validation feedback"
    )
    modes = generate.add_mutually_exclusive_group()
    modes.add_argument(
        "--demo",
        action="store_true",
        help="Run a scripted generation simulation instead of loading a model",
    )
    modes.add_argument("--model", type=Path, help="Local Hugging Face model directory")
    generate.add_argument("--output", type=Path, help="Path for the validated proposal")
    generate.add_argument(
        "--request", default=REQUEST, help="Natural-language request for the proposal"
    )
    generate.add_argument("--max-new-tokens", type=int, default=1024)
    generate.add_argument("--max-attempts", type=int, default=3)
    generate.add_argument("--expected-fold", type=_fold)
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument(
        "--model-revision",
        help="Model revision identifier to record in the optional transcript",
    )
    generate.add_argument(
        "--transcript",
        type=Path,
        help="Write prompts, raw model outputs, and diagnostics as JSONL",
    )
    generate.add_argument(
        "--stream",
        action="store_true",
        help="Write generated text to stderr as each chunk is produced",
    )
    generate.add_argument("--unconstrained", action="store_true")
    generate.add_argument(
        "--examples",
        type=Path,
        default=Path("examples"),
        help="Example directory used only by --demo",
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            result = validate(parse(_read(args.protocol)), args.expected_fold)
            for diagnostic in result.diagnostics:
                print(f"{diagnostic.code} {diagnostic.location}: {diagnostic.message}")
            print(f"Encoded rules: {'PASS' if result.valid else 'FAIL'}")
            print(json.dumps(result.volumes, sort_keys=True))
            return 0 if result.valid else 1
        if args.command == "generate":
            if args.demo:
                if args.output is not None:
                    parser.error("--output cannot be used with --demo")
                if args.transcript is not None:
                    parser.error("--transcript cannot be used with --demo")
                if args.unconstrained:
                    parser.error("--unconstrained cannot be used with --demo")
                return _run_simulation(args.examples)
            if args.model is None:
                parser.error("generate requires --model, or use --demo for the simulation")
            if args.output is None:
                parser.error("generate with --model requires --output")
            if args.transcript is not None and args.transcript.resolve() == args.output.resolve():
                parser.error("--transcript and --output must be different paths")
            text = generate_local(
                args.model,
                request=args.request,
                expected_fold=args.expected_fold,
                constrained=not args.unconstrained,
                max_new_tokens=args.max_new_tokens,
                max_attempts=args.max_attempts,
                seed=args.seed,
                output_path=args.output,
                transcript_path=args.transcript,
                model_revision=args.model_revision,
                stream_output=args.stream,
            )
            _write(args.output, text)
            print(
                f"Semantically validated, unapproved proposal saved to {args.output}; "
                "review it against the original request."
            )
        return 0
    except (
        OSError,
        UnicodeError,
        ProtocolSyntaxError,
        AgentLoopError,
        GenerationError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
