"""Optional real Transformers/Outlines plumbing, without downloaded weights.

The model is tiny and synthesized locally. Scripted scores test integration, not
learned language ability.
"""

import hashlib
import json

import pytest

torch = pytest.importorskip("torch", reason="Install the local extra for Transformers integration")
pytest.importorskip("transformers")

from test_protocol import EXAMPLES
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import (
    AutoModelForCausalLM,
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
)

from outlines_lab.generation import GenerationError, _EosPreference, generate_local

DOCUMENT = (
    'protocol "Test"\nlabware plate96 (max well volume 200 uL)\n\n'
    'step fill\n  add 50 uL "Buffer A" to well A1\n'
)


@pytest.fixture
def tiny_local_model(tmp_path):
    vocabulary = {"<unk>": 0, "<eos>": 1}
    for symbol in pre_tokenizers.ByteLevel.alphabet():
        vocabulary[symbol] = len(vocabulary)
    backend = Tokenizer(models.BPE(vocabulary, [], unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", eos_token="<eos>", pad_token="<eos>"
    )
    tokenizer.save_pretrained(tmp_path)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(tokenizer),
            n_layer=1,
            n_head=1,
            n_embd=16,
            n_positions=4096,
            bos_token_id=1,
            eos_token_id=1,
            pad_token_id=1,
        )
    )
    model.save_pretrained(tmp_path, safe_serialization=True)
    return tmp_path, tokenizer


@pytest.fixture
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_eos_preference_samples_a_complete_document_stop_probability():
    class GrammarMask:
        def reset(self):
            pass

        def __call__(self, input_ids, logits):
            logits[:, 3] = -torch.inf
            return logits

    processor = _EosPreference(GrammarMask(), eos_token_id=1, max_new_tokens=256, torch=torch)
    logits = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    processor(torch.zeros((1, 8), dtype=torch.long), logits)
    adjusted = processor(torch.zeros((1, 136), dtype=torch.long), logits)

    assert torch.softmax(adjusted, dim=-1)[0, 1].item() == pytest.approx(0.75)
    assert torch.isneginf(adjusted[0, 3])


def test_eos_preference_forces_a_stop_near_the_token_limit():
    class GrammarMask:
        def reset(self):
            pass

        def __call__(self, input_ids, logits):
            logits[:, 3] = -torch.inf
            return logits

    processor = _EosPreference(GrammarMask(), eos_token_id=1, max_new_tokens=65, torch=torch)
    processor(torch.zeros((1, 8), dtype=torch.long), torch.zeros((1, 4)))
    adjusted = processor(torch.zeros((1, 9), dtype=torch.long), torch.zeros((1, 4)))

    assert torch.isneginf(adjusted[0, 0])
    assert torch.isneginf(adjusted[0, 2])
    assert adjusted[0, 1] == 0


def _script_model_scores(monkeypatch, tokenizer, text, illegal_high_score=False):
    original = AutoModelForCausalLM.from_pretrained
    proposed = tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]
    illegal = tokenizer.encode("\r", add_special_tokens=False)[0]

    def load(*args, **kwargs):
        model = original(*args, **kwargs)
        position = 0

        def set_scores(module, inputs, output):
            nonlocal position
            output.logits.fill_(-10_000)
            output.logits[:, -1, proposed[position]] = 10_000
            if illegal_high_score:
                output.logits[:, -1, illegal] = 20_000
            position += 1
            return output

        model.register_forward_hook(set_scores)
        return model

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", load)


@pytest.mark.parametrize("constrained", [True, False])
def test_local_generation_complete_document(
    tiny_local_model, single_thread, monkeypatch, constrained
):
    path, tokenizer = tiny_local_model
    _script_model_scores(monkeypatch, tokenizer, DOCUMENT, illegal_high_score=constrained)
    assert generate_local(path, constrained=constrained, max_new_tokens=256) == DOCUMENT


@pytest.mark.parametrize("constrained", [True, False])
def test_local_generation_can_stream_response_to_stderr(
    tiny_local_model, single_thread, monkeypatch, capsys, constrained
):
    path, tokenizer = tiny_local_model
    _script_model_scores(monkeypatch, tokenizer, DOCUMENT, illegal_high_score=constrained)

    result = generate_local(
        path,
        constrained=constrained,
        max_new_tokens=256,
        stream_output=True,
    )

    captured = capsys.readouterr()
    assert result == DOCUMENT
    assert captured.out == ""
    assert captured.err == f"Attempt 1: {DOCUMENT}\n"


def test_local_generation_rejects_token_budget_exhaustion(
    tiny_local_model, single_thread, monkeypatch
):
    path, tokenizer = tiny_local_model
    _script_model_scores(monkeypatch, tokenizer, DOCUMENT)
    with pytest.raises(GenerationError, match="may contain malformed DSL.*or may be truncated"):
        generate_local(path, max_new_tokens=4, max_attempts=1)


def test_local_unconstrained_generation_rejects_prose(tiny_local_model, single_thread, monkeypatch):
    path, tokenizer = tiny_local_model
    _script_model_scores(monkeypatch, tokenizer, "Here is your protocol.")
    with pytest.raises(GenerationError, match="Unconstrained agent loop failed"):
        generate_local(path, constrained=False, max_new_tokens=256, max_attempts=1)


def test_constrained_generation_requires_eos_policy_processor(
    tiny_local_model, single_thread, monkeypatch
):
    import outlines

    path, _ = tiny_local_model

    class GeneratorWithoutProcessor:
        def __init__(self, model, constraint, backend):
            assert constraint is not None
            assert backend == "llguidance"

    monkeypatch.setattr(outlines, "Generator", GeneratorWithoutProcessor)

    with pytest.raises(GenerationError, match="requires a logits processor"):
        generate_local(path, max_attempts=1)


def test_unconstrained_syntax_retry_records_jsonl_transcript(
    tiny_local_model, single_thread, monkeypatch, tmp_path
):
    import outlines

    path, _ = tiny_local_model
    proposals = ["Here is your protocol.", DOCUMENT]
    prompts = []

    class ScriptedGenerator:
        def __init__(self, model, constraint, backend):
            assert constraint is None
            assert backend == "llguidance"

        def __call__(self, prompt, **kwargs):
            prompts.append(prompt)
            return proposals[len(prompts) - 1]

    monkeypatch.setattr(outlines, "Generator", ScriptedGenerator)
    transcript = tmp_path / "attempts.jsonl"

    result = generate_local(
        path,
        constrained=False,
        expected_fold=2,
        max_attempts=2,
        transcript_path=transcript,
        model_revision="test-revision",
    )

    records = [json.loads(line) for line in transcript.read_text().splitlines()]
    assert result == DOCUMENT
    assert len(records) == 3
    header = records[0]
    assert header["record_type"] == "run"
    assert header["outcome"] == "success"
    assert header["model_revision"] == "test-revision"
    expected_hashes = [
        {
            "file": str(weight.relative_to(path)),
            "sha256": hashlib.sha256(weight.read_bytes()).hexdigest(),
        }
        for weight in sorted(path.rglob("*.safetensors"))
    ]
    assert header["weights_sha256"] == expected_hashes
    assert header["constrained"] is False
    assert header["seed"] == 0
    assert header["do_sample"] is True
    assert header["eos_stop_probability"] is None
    assert header["max_new_tokens"] == 1024
    assert header["max_attempts"] == 2
    assert header["expected_fold"] == 2
    assert set(header["package_versions"]) == {
        "outlines-lab",
        "outlines",
        "llguidance",
        "transformers",
        "torch",
    }
    attempt_records = records[1:]
    assert all(record["record_type"] == "attempt" for record in attempt_records)
    assert attempt_records[0]["prompt"] == prompts[0]
    assert attempt_records[0]["raw_output"] == proposals[0]
    assert attempt_records[0]["status"] == "syntax_error"
    assert attempt_records[0]["diagnostics"][0]["code"] == "SYNTAX"
    assert attempt_records[0]["diagnostics"][0]["location"].startswith("line 1, column")
    assert attempt_records[1]["prompt"] == prompts[1]
    assert proposals[0] in attempt_records[1]["prompt"]
    assert "SYNTAX" in attempt_records[1]["prompt"]
    assert "(50 + 50) / 50 = 2" in attempt_records[1]["prompt"]
    assert attempt_records[1]["raw_output"] == DOCUMENT
    assert attempt_records[1]["status"] == "passed"


def test_failed_generation_still_writes_all_raw_attempts(
    tiny_local_model, single_thread, monkeypatch, tmp_path
):
    import outlines

    path, _ = tiny_local_model
    proposals = ["not a DSL document", "still not a DSL document"]
    prompts = []

    class ScriptedGenerator:
        def __init__(self, model, constraint, backend):
            assert constraint is None

        def __call__(self, prompt, **kwargs):
            prompts.append(prompt)
            return proposals[len(prompts) - 1]

    monkeypatch.setattr(outlines, "Generator", ScriptedGenerator)
    transcript = tmp_path / "failed-attempts.jsonl"

    with pytest.raises(GenerationError, match="No valid proposal after 2 attempts"):
        generate_local(
            path,
            constrained=False,
            max_attempts=2,
            transcript_path=transcript,
            model_revision="failed-run-revision",
        )

    records = [json.loads(line) for line in transcript.read_text().splitlines()]
    assert records[0]["record_type"] == "run"
    assert records[0]["outcome"] == "retry_limit_exhausted"
    assert records[0]["model_revision"] == "failed-run-revision"
    attempt_records = records[1:]
    assert [record["raw_output"] for record in attempt_records] == proposals
    assert all(record["status"] == "syntax_error" for record in attempt_records)
    assert all(record["diagnostics"][0]["code"] == "SYNTAX" for record in attempt_records)
    assert attempt_records[0]["prompt"] == prompts[0]
    assert attempt_records[1]["prompt"] == prompts[1]
    assert "line 1, column" in attempt_records[1]["prompt"]
    assert "^" in attempt_records[1]["prompt"]


def test_token_limited_output_is_labeled_and_truncated_in_retry_prompt(
    tiny_local_model, single_thread, monkeypatch, tmp_path
):
    import outlines

    path, _ = tiny_local_model
    first_proposal = "not a DSL document " + ("0" * 3000) + "TAIL_SENTINEL"
    proposals = [first_proposal, DOCUMENT]
    prompts = []

    class ScriptedGenerator:
        def __init__(self, model, constraint, backend):
            assert constraint is None

        def __call__(self, prompt, **kwargs):
            prompts.append(prompt)
            return proposals[len(prompts) - 1]

    monkeypatch.setattr(outlines, "Generator", ScriptedGenerator)
    transcript = tmp_path / "token-limited-attempts.jsonl"

    assert (
        generate_local(
            path,
            constrained=False,
            max_new_tokens=16,
            max_attempts=2,
            transcript_path=transcript,
        )
        == DOCUMENT
    )

    records = [json.loads(line) for line in transcript.read_text().splitlines()]
    failed_attempt = records[1]
    retry_attempt = records[2]
    assert failed_attempt["diagnostics"][0]["code"] == "MAX_NEW_TOKENS"
    assert "Output cut off at max_new_tokens=16" in failed_attempt["diagnostics"][0]["message"]
    assert first_proposal not in retry_attempt["prompt"]
    assert first_proposal[:1000] in retry_attempt["prompt"]
    assert "[... truncated ...]" in retry_attempt["prompt"]
    assert "TAIL_SENTINEL" not in retry_attempt["prompt"]
    assert "MAX_NEW_TOKENS" in retry_attempt["prompt"]


def test_local_generation_includes_semantic_diagnostics_in_retry_prompt(
    tiny_local_model, single_thread, monkeypatch, capsys, tmp_path
):
    import outlines

    path, _ = tiny_local_model
    candidates = [
        (EXAMPLES / f"{name}.lab").read_text()
        for name in ("01_overflow", "02_wrong_intent", "03_two_fold")
    ]
    prompts = []

    class ScriptedGenerator:
        def __init__(self, model, constraint, backend):
            assert constraint is not None
            assert backend == "llguidance"
            self.logits_processor = type("Processor", (), {"reset": lambda self: None})()

        def __call__(self, prompt, **kwargs):
            prompts.append(prompt)
            return candidates[len(prompts) - 1]

    monkeypatch.setattr(outlines, "Generator", ScriptedGenerator)
    transcript = tmp_path / "constrained-attempts.jsonl"
    result = generate_local(
        path,
        expected_fold=2,
        max_attempts=3,
        transcript_path=transcript,
    )

    assert result == candidates[2]
    run_record = json.loads(transcript.read_text().splitlines()[0])
    assert run_record["eos_policy_installed"] is True
    assert run_record["eos_stop_probability"] == 0.75
    assert run_record["temperature"] == 1.0
    assert run_record["top_k"] == 0
    assert run_record["top_p"] == 1.0
    assert len(prompts) == 3
    assert "INTENT_FOLD" in prompts[1]
    assert "CAPACITY" in prompts[1]
    assert "(50 + 50) / 50 = 2" in prompts[0]
    assert "INTENT_FOLD" in prompts[2]
    assert candidates[0] in prompts[1]
    assert candidates[1] in prompts[2]
    captured = capsys.readouterr()
    assert "Retrying attempt 2/3: the previous proposal failed semantic validation" in captured.err
    assert "CAPACITY" in captured.err
    assert "INTENT_FOLD" in captured.err
    assert (
        "The previous proposal (up to 1,000 characters) and its diagnostics are "
        "included in the retry prompt."
    ) in captured.err
    assert "Retrying attempt 3/3" in captured.err
