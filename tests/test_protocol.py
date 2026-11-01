from fractions import Fraction
from pathlib import Path

import pytest

from outlines_lab.language import ProtocolSyntaxError, parse
from outlines_lab.validation import validate

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture
def two_fold():
    return (EXAMPLES / "03_two_fold.lab").read_text()


@pytest.mark.parametrize("name", ["01_overflow", "02_wrong_intent", "03_two_fold"])
def test_examples_parse(name):
    assert parse((EXAMPLES / f"{name}.lab").read_text()).capacity == 200


def test_volume_limit():
    result = validate(parse((EXAMPLES / "01_overflow.lab").read_text()))
    assert [d.code for d in result.diagnostics] == ["CAPACITY"] * 6
    assert result.volumes == {**{f"A{i}": 210 for i in range(1, 6)}, "A6": 220}


def test_valid_syntax_and_rules_do_not_imply_intent():
    protocol = parse((EXAMPLES / "02_wrong_intent.lab").read_text())
    assert validate(protocol).valid
    result = validate(protocol, expected_fold=2)
    assert [d.code for d in result.diagnostics] == ["INTENT_FOLD"] * 5
    assert all("10-fold" in d.message for d in result.diagnostics)


def test_two_fold_volumes_and_final_concentrations(two_fold):
    result = validate(parse(two_fold), expected_fold=2)
    assert result.valid
    assert result.volumes == {"A1": 150, "A2": 150, "A3": 150, "A4": 150, "A5": 150, "A6": 200}
    # Reagent addition and retention at A6 alter final concentration ratios.
    assert result.sample_fraction == {
        "A1": Fraction(1, 3),
        "A2": Fraction(1, 6),
        "A3": Fraction(1, 12),
        "A4": Fraction(1, 24),
        "A5": Fraction(1, 48),
        "A6": Fraction(1, 64),
    }


@pytest.mark.parametrize(
    "old,new,code",
    [
        ("add 50 uL", "add 0 uL", "VOLUME"),
        ("transfer 50 uL", "transfer 150 uL", "UNDERFLOW"),
        ("mixing 3", "mixing 0", "TRANSFER"),
        ("from A1 to A6", "from A1 to A1", "TRANSFER"),
        ("wells A2-A6", "wells A6-A2", "RANGE"),
        ("wells A2-A6", "wells A2-B6", "RANGE"),
        ("step read", "step develop", "DUPLICATE_STEP"),
        ("incubate 15", "incubate 0", "DURATION"),
        ("at 450 nm", "at 0 nm", "WAVELENGTH"),
        ("nm in wells A1-A6", "nm in well H12", "EMPTY_READ"),
        ("volume 200 uL", "volume 201 uL", "CAPACITY_LIMIT"),
    ],
)
def test_semantic_errors_are_not_syntax_errors(two_fold, old, new, code):
    result = validate(parse(two_fold.replace(old, new)))
    assert code in [d.code for d in result.diagnostics]


def test_transient_overflow_is_caught():
    text = (
        'protocol "Transient"\nlabware plate96 (max well volume 200 uL)\n\n'
        'step dilute\n  add 200 uL "Buffer A" to wells A1-A3\n'
        "  serial transfer 50 uL from A1 to A3, mixing 3 times at each well\n"
    )
    result = validate(parse(text))
    assert result.volumes["A2"] == 200
    assert any("A2: 250" in d.message for d in result.diagnostics)


@pytest.mark.parametrize(
    "old,new",
    [
        ("uL", "mL"),
        ("well A1", "well A13"),
        ("well A1", "well I1"),
        ("add 50", "add -50"),
        ("add 50", "add 1.5"),
        ("step dilute", "step Dilute"),
        ('"Buffer A"', '"Buffer\\A"'),
    ],
)
def test_malformed_documents_rejected(two_fold, old, new):
    with pytest.raises(ProtocolSyntaxError):
        parse(two_fold.replace(old, new))


@pytest.mark.parametrize(
    "text,accepted",
    [
        ('protocol ""', False),
        (f'protocol "{"a" * 60}"', True),
        (f'protocol "{"a" * 61}"', False),
        ("max well volume 999999 uL", True),
        ("max well volume 1000000 uL", False),
    ],
)
def test_string_and_integer_length_bounds(two_fold, text, accepted):
    if text.startswith("protocol "):
        candidate = text + two_fold[two_fold.index("\n") :]
    else:
        candidate = two_fold.replace("max well volume 200 uL", text, 1)

    if accepted:
        parse(candidate)
    else:
        with pytest.raises(ProtocolSyntaxError):
            parse(candidate)


def test_incomplete_and_prose_rejected(two_fold):
    for text in (
        two_fold[:-8],
        "Here you go:\n" + two_fold,
        (EXAMPLES / "00_malformed.lab").read_text(),
    ):
        with pytest.raises(ProtocolSyntaxError):
            parse(text)


def test_invalid_fold_parameter(two_fold):
    with pytest.raises(ValueError, match="at least 2"):
        validate(parse(two_fold), expected_fold=1)
