"""Bounded generation, semantic-validation, and diagnostic-feedback loop."""

from collections.abc import Callable
from dataclasses import dataclass

from .language import ProtocolSyntaxError, parse
from .validation import Diagnostic, Validation, validate


@dataclass(frozen=True)
class AgentPrompt:
    request: str
    expected_fold: int | None
    attempt: int
    previous_proposal: str | None
    diagnostics: tuple[Diagnostic, ...]


@dataclass(frozen=True)
class AgentAttempt:
    proposal: str
    prompt: AgentPrompt
    validation: Validation | None
    diagnostics: tuple[Diagnostic, ...]


@dataclass(frozen=True)
class AgentResult:
    proposal: str
    validation: Validation
    attempts: tuple[AgentAttempt, ...]


class AgentLoopError(ValueError):
    """The agent could not produce a syntactically and semantically valid proposal."""

    def __init__(self, message: str, attempts: tuple[AgentAttempt, ...]):
        super().__init__(message)
        self.attempts = attempts


def run_agent_loop(
    propose: Callable[[AgentPrompt], str],
    *,
    request: str,
    expected_fold: int | None = None,
    max_attempts: int = 3,
    truncation_diagnostic: Callable[[str], Diagnostic | None] | None = None,
) -> AgentResult:
    if not request.strip():
        raise ValueError("The user request must be nonempty")
    if max_attempts <= 0:
        raise ValueError("Maximum attempts must be positive")
    if expected_fold is not None and expected_fold < 2:
        raise ValueError("Expected dilution fold must be at least 2")

    attempts: list[AgentAttempt] = []
    previous_proposal: str | None = None
    diagnostics: tuple[Diagnostic, ...] = ()
    for attempt_number in range(1, max_attempts + 1):
        context = AgentPrompt(
            request=request,
            expected_fold=expected_fold,
            attempt=attempt_number,
            previous_proposal=previous_proposal,
            diagnostics=diagnostics,
        )
        proposal = propose(context)
        cutoff = truncation_diagnostic(proposal) if truncation_diagnostic else None
        try:
            protocol = parse(proposal)
        except ProtocolSyntaxError as exc:
            location = (
                f"line {exc.line}, column {exc.column}"
                if exc.line is not None and exc.column is not None
                else "document"
            )
            diagnostic = Diagnostic("SYNTAX", location, exc.context or str(exc))
            attempt_diagnostics = (diagnostic,) if cutoff is None else (cutoff, diagnostic)
            attempts.append(AgentAttempt(proposal, context, None, attempt_diagnostics))
            previous_proposal = proposal
            diagnostics = attempt_diagnostics
            continue

        validation = validate(protocol, expected_fold=expected_fold)
        attempt_diagnostics = (
            validation.diagnostics if cutoff is None else (cutoff, *validation.diagnostics)
        )
        current = AgentAttempt(proposal, context, validation, attempt_diagnostics)
        attempts.append(current)
        if validation.valid:
            return AgentResult(proposal, validation, tuple(attempts))

        previous_proposal = proposal
        diagnostics = attempt_diagnostics

    summary = "; ".join(f"{item.code} {item.location}: {item.message}" for item in diagnostics)
    raise AgentLoopError(
        f"No valid proposal after {max_attempts} attempts. Last diagnostics: {summary}",
        tuple(attempts),
    )
