import pytest
from test_protocol import EXAMPLES

from outlines_lab.agent import AgentLoopError, AgentPrompt, run_agent_loop
from outlines_lab.generation import masked_replay
from outlines_lab.validation import Diagnostic


def test_agent_feeds_semantic_diagnostics_back_before_handoff():
    candidates = {
        "initial": (EXAMPLES / "01_overflow.lab").read_text(),
        "capacity": (EXAMPLES / "02_wrong_intent.lab").read_text(),
        "intent": (EXAMPLES / "03_two_fold.lab").read_text(),
    }
    prompts: list[AgentPrompt] = []

    def propose(context: AgentPrompt) -> str:
        prompts.append(context)
        codes = {diagnostic.code for diagnostic in context.diagnostics}
        if not codes:
            candidate = candidates["initial"]
        elif "CAPACITY" in codes:
            candidate = candidates["capacity"]
        elif "INTENT_FOLD" in codes:
            candidate = candidates["intent"]
        else:
            pytest.fail(f"Unexpected diagnostics: {codes}")
        return masked_replay(candidate).text

    result = run_agent_loop(
        propose,
        request="Make a two-fold dilution series with a 200 uL capacity.",
        expected_fold=2,
    )

    assert len(result.attempts) == 3
    assert not result.attempts[0].validation.valid
    assert {item.code for item in result.attempts[0].validation.diagnostics} == {
        "CAPACITY",
        "INTENT_FOLD",
    }
    assert {item.code for item in result.attempts[1].validation.diagnostics} == {
        "INTENT_FOLD",
    }
    assert result.attempts[2].validation.valid
    assert prompts[0].diagnostics == ()
    assert {item.code for item in prompts[1].diagnostics} == {"CAPACITY", "INTENT_FOLD"}
    assert {item.code for item in prompts[2].diagnostics} == {"INTENT_FOLD"}
    assert all(prompt.expected_fold == 2 for prompt in prompts)
    assert prompts[1].previous_proposal == candidates["initial"]
    assert result.proposal == candidates["intent"]


def test_agent_fails_explicitly_when_attempt_limit_is_reached():
    candidate = (EXAMPLES / "02_wrong_intent.lab").read_text()

    with pytest.raises(AgentLoopError, match="after 2 attempts") as error:
        run_agent_loop(
            lambda context: masked_replay(candidate).text,
            request="Make a two-fold dilution series.",
            expected_fold=2,
            max_attempts=2,
        )

    assert len(error.value.attempts) == 2
    assert all(not attempt.validation.valid for attempt in error.value.attempts)


def test_agent_records_syntax_failure_and_retries_with_lark_diagnostic():
    valid = (EXAMPLES / "03_two_fold.lab").read_text()
    proposals = iter(["Here is your protocol.", valid])
    prompts: list[AgentPrompt] = []

    def propose(context: AgentPrompt) -> str:
        prompts.append(context)
        return next(proposals)

    result = run_agent_loop(
        propose,
        request="Make a two-fold dilution series.",
        expected_fold=2,
    )

    failed = result.attempts[0]
    assert failed.validation is None
    assert failed.proposal == "Here is your protocol."
    assert failed.diagnostics[0].code == "SYNTAX"
    assert "line 1, column" in failed.diagnostics[0].location
    assert "^" in failed.diagnostics[0].message
    assert prompts[1].previous_proposal == failed.proposal
    assert prompts[1].diagnostics == failed.diagnostics
    assert result.proposal == valid


def test_agent_labels_token_limited_syntax_failure():
    valid = (EXAMPLES / "03_two_fold.lab").read_text()
    proposals = iter(["incomplete output", valid])
    prompts: list[AgentPrompt] = []

    def propose(context: AgentPrompt) -> str:
        prompts.append(context)
        return next(proposals)

    result = run_agent_loop(
        propose,
        request="Make a two-fold dilution series.",
        expected_fold=2,
        truncation_diagnostic=lambda _: Diagnostic(
            "MAX_NEW_TOKENS", "generation", "Output cut off at max_new_tokens=8."
        ),
    )

    assert result.proposal == valid
    assert [item.code for item in result.attempts[0].diagnostics] == [
        "MAX_NEW_TOKENS",
        "SYNTAX",
    ]
    assert "Output cut off at max_new_tokens=8." in result.attempts[0].diagnostics[0].message
    assert prompts[1].diagnostics == result.attempts[0].diagnostics


def test_agent_labels_token_limited_semantic_failure():
    overflow = (EXAMPLES / "01_overflow.lab").read_text()
    valid = (EXAMPLES / "03_two_fold.lab").read_text()
    proposals = iter([overflow, valid])

    def truncation_diagnostic(proposal):
        if proposal == overflow:
            return Diagnostic("MAX_NEW_TOKENS", "generation", "Output cut off at max_new_tokens=8.")
        return None

    result = run_agent_loop(
        lambda _: next(proposals),
        request="Make a two-fold dilution series.",
        expected_fold=2,
        truncation_diagnostic=truncation_diagnostic,
    )

    assert result.proposal == valid
    codes = [item.code for item in result.attempts[0].diagnostics]
    assert codes[0] == "MAX_NEW_TOKENS"
    assert "CAPACITY" in codes
    assert "INTENT_FOLD" in codes
    assert result.attempts[1].prompt.diagnostics == result.attempts[0].diagnostics


@pytest.mark.parametrize(
    ("user_request", "expected_fold", "max_attempts"),
    [("", None, 1), ("request", 1, 1), ("request", None, 0)],
)
def test_agent_rejects_invalid_configuration(user_request, expected_fold, max_attempts):
    with pytest.raises(ValueError):
        run_agent_loop(
            lambda _: pytest.fail("Proposer must not be called"),
            request=user_request,
            expected_fold=expected_fold,
            max_attempts=max_attempts,
        )
