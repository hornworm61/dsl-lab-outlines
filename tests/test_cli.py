from types import SimpleNamespace
from pathlib import Path

import pytest

import outlines_lab.cli as cli
from outlines_lab.cli import main

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_generate_demo_shows_semantic_feedback_before_review_handoff(capsys):
    assert main(["generate", "--demo", "--examples", str(EXAMPLES)]) == 0
    output = capsys.readouterr().out
    assert "SIMULATION ONLY: scripted proposals, not an actual language model." in output
    assert "Attempt 1: grammar PASS; semantic validation FAIL." in output
    assert "Attempt 2: grammar PASS; semantic validation FAIL." in output
    assert "Attempt 3: grammar PASS; semantic validation PASS." in output
    assert "Feeding these diagnostics into the next generation prompt." in output
    assert "after 3 attempts" in output
    assert "human for intent review" in output
    assert "scripted logits, not an LLM" in output


def test_generate_demo_handles_syntax_failure(monkeypatch, capsys):
    def run_with_syntax_failure(propose, **kwargs):
        propose(SimpleNamespace(diagnostics=()))
        validation = SimpleNamespace(valid=True, diagnostics=(), volumes={})
        propose(SimpleNamespace(diagnostics=(SimpleNamespace(code="CAPACITY"),)))
        return SimpleNamespace(
            attempts=(
                SimpleNamespace(validation=None),
                SimpleNamespace(validation=validation),
            ),
            validation=validation,
        )

    monkeypatch.setattr(cli, "run_agent_loop", run_with_syntax_failure)

    assert cli._run_simulation(EXAMPLES) == 0
    output = capsys.readouterr().out
    assert "Attempt 1: grammar FAIL; semantic validation NOT RUN." in output
    assert "Attempt 2: grammar PASS; semantic validation PASS." in output


def test_generate_requires_model_unless_demo(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["generate"])

    assert exc.value.code == 2
    assert "generate requires --model, or use --demo" in capsys.readouterr().err


def test_validate_cli_checks_explicit_semantic_requirement(capsys):
    wrong_fold = str(EXAMPLES / "02_wrong_intent.lab")
    assert main(["validate", wrong_fold]) == 0
    capsys.readouterr()
    assert main(["validate", wrong_fold, "--expected-fold", "2"]) == 1
    output = capsys.readouterr().out
    assert "INTENT_FOLD" in output
    assert "Encoded rules: FAIL" in output


@pytest.mark.parametrize("name", ["00_malformed", "01_overflow"])
def test_validate_cli_reports_syntax_and_semantic_failures(name, capsys):
    result = main(["validate", str(EXAMPLES / f"{name}.lab")])
    assert result == (2 if name == "00_malformed" else 1)
    captured = capsys.readouterr()
    output = captured.err if name == "00_malformed" else captured.out
    expected = "ERROR:" if name == "00_malformed" else "Encoded rules: FAIL"
    assert expected in output


def test_cli_exposes_only_core_commands(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])

    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "generate" in help_text
    assert "validate" in help_text
    assert "demo" not in help_text

    with pytest.raises(SystemExit) as exc:
        main(["generate", "--help"])

    assert exc.value.code == 0
    generate_help = capsys.readouterr().out
    assert "--demo" in generate_help
    assert "--model" in generate_help
    assert "--output" in generate_help
    assert "--transcript" in generate_help
    assert "--model-revision" in generate_help
    assert "--stream" in generate_help
