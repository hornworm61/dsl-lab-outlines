import pytest
from test_protocol import EXAMPLES

from outlines_lab.generation import GenerationError, generate_local, masked_replay
from outlines_lab.language import parse
from outlines_lab.validation import validate


@pytest.mark.parametrize("name", ["01_overflow", "02_wrong_intent", "03_two_fold"])
def test_real_outlines_masks_illegal_tokens_and_allows_semantic_errors(name):
    original = (EXAMPLES / f"{name}.lab").read_text()
    trace = masked_replay(original)
    assert trace.text == original
    assert trace.blocked_tokens == len(original) + 1
    assert trace.token_count == len(original) + 1
    assert validate(parse(trace.text)).valid == (name != "01_overflow")


@pytest.mark.parametrize("text", ["Here is your protocol:", 'protocol "Incomplete"', "bad\r\n"])
def test_mask_rejects_bad_prefix_and_premature_eos(text):
    with pytest.raises(GenerationError, match="rejected|not masked"):
        masked_replay(text)


def test_missing_local_model_is_explicit(tmp_path):
    with pytest.raises(GenerationError, match="existing local directory"):
        generate_local(tmp_path / "missing")


def test_invalid_budget(tmp_path):
    with pytest.raises(GenerationError, match="positive"):
        generate_local(tmp_path, max_new_tokens=0)


def test_transcript_path_must_not_already_exist(tmp_path):
    transcript = tmp_path / "attempts.jsonl"
    transcript.touch()
    with pytest.raises(GenerationError, match="already exists"):
        generate_local(tmp_path, transcript_path=transcript)


def test_output_path_must_be_free_before_model_loading(tmp_path):
    output = tmp_path / "proposal.lab"
    output.touch()

    with pytest.raises(GenerationError, match="Output path already exists"):
        generate_local(tmp_path, output_path=output)


def test_output_and_transcript_paths_must_be_different(tmp_path):
    output = tmp_path / "proposal.lab"

    with pytest.raises(GenerationError, match="must be different"):
        generate_local(tmp_path, output_path=output, transcript_path=output)


@pytest.mark.parametrize("well", ["A10", "G11", "H12"])
def test_two_digit_wells_work_in_both_parsers(well):
    text = (
        'protocol "Two digit well"\nlabware plate96 (max well volume 200 uL)\n\n'
        f'step fill\n  add 50 uL "Buffer A" to well {well}\n'
    )
    trace = masked_replay(text)
    assert validate(parse(trace.text)).volumes == {well: 50}
