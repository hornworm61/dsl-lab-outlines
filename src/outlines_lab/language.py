"""The same grammar is used for parsing and Outlines CFG masking."""

from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files

from lark import Lark, Transformer, UnexpectedInput, v_args
from lark.exceptions import VisitError


def grammar() -> str:
    return files("outlines_lab").joinpath("protocol.lark").read_text(encoding="utf-8")


class ProtocolSyntaxError(ValueError):
    """The document does not conform to the supported DSL."""

    def __init__(
        self,
        message: str,
        *,
        line: int | None = None,
        column: int | None = None,
        context: str | None = None,
    ) -> None:
        super().__init__(message)
        self.line = line
        self.column = column
        self.context = context


@dataclass(frozen=True)
class Targets:
    first: str
    last: str


@dataclass(frozen=True)
class Add:
    volume: int
    reagent: str
    targets: Targets


@dataclass(frozen=True)
class Transfer:
    volume: int
    first: str
    last: str
    mixing: int


@dataclass(frozen=True)
class Incubate:
    minutes: int
    temperature: int


@dataclass(frozen=True)
class Measure:
    wavelength: int
    targets: Targets


Command = Add | Transfer | Incubate | Measure


@dataclass(frozen=True)
class Step:
    name: str
    commands: tuple[Command, ...]


@dataclass(frozen=True)
class Protocol:
    name: str
    capacity: int
    steps: tuple[Step, ...]


@v_args(inline=True)
class _Build(Transformer):
    def STRING(self, token):
        return str(token)[1:-1]

    def INT(self, token):
        return int(token)

    def WELL(self, token):
        return str(token)

    def NAME(self, token):
        return str(token)

    def single(self, well):
        return Targets(well, well)

    def span(self, first, last):
        return Targets(first, last)

    def add(self, volume, reagent, targets):
        return Add(volume, reagent, targets)

    def transfer(self, volume, first, last, mixing):
        return Transfer(volume, first, last, mixing)

    def incubate(self, minutes, temperature):
        return Incubate(minutes, temperature)

    def measure(self, wavelength, targets):
        return Measure(wavelength, targets)

    def step(self, name, *commands):
        return Step(name, commands)

    def start(self, name, capacity, *steps):
        return Protocol(name, capacity, steps)


@lru_cache(maxsize=1)
def _parser() -> Lark:
    return Lark(grammar(), parser="lalr")


def parse(text: str) -> Protocol:
    try:
        tree = _parser().parse(text)
        return _Build().transform(tree)
    except UnexpectedInput as exc:
        context = exc.get_context(text, span=45)
        raise ProtocolSyntaxError(
            f"Syntax error at line {exc.line}, column {exc.column}:\n{context}",
            line=exc.line,
            column=exc.column,
            context=context,
        ) from exc
    except VisitError as exc:
        raise ProtocolSyntaxError(f"Cannot construct protocol: {exc.orig_exc}") from exc
