"""A small, explicit volume model, not an equipment safety validator."""

from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise

from .language import Add, Incubate, Measure, Protocol, Targets, Transfer


@dataclass(frozen=True)
class Diagnostic:
    code: str
    location: str
    message: str


@dataclass(frozen=True)
class Validation:
    diagnostics: tuple[Diagnostic, ...]
    volumes: dict[str, int]
    sample_fraction: dict[str, Fraction]

    @property
    def valid(self) -> bool:
        return not self.diagnostics


def wells(targets: Targets) -> tuple[str, ...]:
    if targets.first[0] != targets.last[0]:
        raise ValueError("Ranges must stay within one row")
    first, last = int(targets.first[1:]), int(targets.last[1:])
    if first > last:
        raise ValueError("Ranges must be ascending")
    return tuple(f"{targets.first[0]}{column}" for column in range(first, last + 1))


def validate(protocol: Protocol, expected_fold: int | None = None) -> Validation:
    if expected_fold is not None and expected_fold < 2:
        raise ValueError("Expected dilution fold must be at least 2")
    volumes: dict[str, int] = {}
    sample: dict[str, Fraction] = {}
    diagnostics: list[Diagnostic] = []
    seen: set[str] = set()

    def error(code: str, location: str, message: str) -> None:
        diagnostics.append(Diagnostic(code, location, message))

    def capacity(well: str, location: str) -> None:
        if volumes[well] > protocol.capacity:
            error(
                "CAPACITY", location, f"{well}: {volumes[well]} uL exceeds {protocol.capacity} uL"
            )

    if protocol.capacity <= 0 or protocol.capacity > 200:
        error("CAPACITY_LIMIT", "labware", "Illustrative plate96 capacity must be 1..200 uL")
    for step in protocol.steps:
        if step.name in seen:
            error("DUPLICATE_STEP", step.name, "Step names must be unique")
        seen.add(step.name)
        for index, command in enumerate(step.commands, 1):
            location = f"{step.name}:{index}"
            if isinstance(command, (Add, Measure, Transfer)):
                targets = (
                    Targets(command.first, command.last)
                    if isinstance(command, Transfer)
                    else command.targets
                )
                try:
                    selected = wells(targets)
                except ValueError as exc:
                    error("RANGE", location, str(exc))
                    continue
            if isinstance(command, Add):
                if command.volume <= 0:
                    error("VOLUME", location, "Addition volume must be positive")
                    continue
                # Only this exact illustrative label contributes tracked sample mass.
                for well in selected:
                    volumes[well] = volumes.get(well, 0) + command.volume
                    sample[well] = sample.get(well, Fraction(0)) + (
                        command.volume if command.reagent == "Sample S1" else 0
                    )
                    capacity(well, location)
            elif isinstance(command, Transfer):
                if command.volume <= 0 or command.mixing <= 0 or len(selected) < 2:
                    error(
                        "TRANSFER",
                        location,
                        "Transfer needs positive volume, positive mixing, and distinct endpoints",
                    )
                    continue
                for source, target in pairwise(selected):
                    available = volumes.get(source, 0)
                    if available < command.volume:
                        error(
                            "UNDERFLOW",
                            location,
                            f"{source}: cannot transfer {command.volume} uL from {available} uL",
                        )
                        break
                    receiver = volumes.get(target, 0)
                    if expected_fold is not None:
                        actual = Fraction(receiver + command.volume, command.volume)
                        if actual != expected_fold:
                            error(
                                "INTENT_FOLD",
                                location,
                                f"{source}->{target}: {actual}-fold, expected {expected_fold}-fold",
                            )
                    mass = sample.get(source, Fraction(0)) * command.volume / available
                    volumes[source] = available - command.volume
                    sample[source] = sample.get(source, Fraction(0)) - mass
                    volumes[target] = receiver + command.volume
                    sample[target] = sample.get(target, Fraction(0)) + mass
                    # Check transient receiving volume, not just final volume.
                    capacity(target, location)
            elif isinstance(command, Incubate):
                if command.minutes <= 0:
                    error("DURATION", location, "Incubation duration must be positive")
            elif isinstance(command, Measure):
                if command.wavelength <= 0:
                    error("WAVELENGTH", location, "Wavelength must be positive")
                for well in selected:
                    if volumes.get(well, 0) == 0:
                        error("EMPTY_READ", location, f"{well}: cannot read an empty well")
    fractions = {
        well: sample.get(well, Fraction(0)) / volume if volume else Fraction(0)
        for well, volume in volumes.items()
    }
    return Validation(tuple(diagnostics), dict(sorted(volumes.items())), fractions)
