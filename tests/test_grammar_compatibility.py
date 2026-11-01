"""Finite differential evidence, not a proof of language equivalence."""

import random

import llguidance
import numpy as np
import pytest
from lark import Lark, UnexpectedInput
from outlines.backends.llguidance import LLGuidanceLogitsProcessor
from test_protocol import EXAMPLES

from outlines_lab.generation import _character_processor
from outlines_lab.language import ProtocolSyntaxError, grammar, parse
from outlines_lab.validation import validate


@pytest.fixture(scope="module")
def parsers():
    return (
        Lark(grammar(), parser="lalr"),
        Lark(grammar(), parser="earley", lexer="dynamic_complete", ambiguity="explicit"),
    )


def _accepts(parser, text):
    try:
        tree = parser.parse(text)
    except UnexpectedInput:
        return False
    assert not list(tree.find_data("_ambig")), repr(text)
    return True


def _check_document(text, parsers, tokenizer=None):
    lalr, earley = parsers
    if tokenizer is None:
        characters, processor = _character_processor()
        proposed = characters(text)
        eos = characters.eos_token_id
        vocab_size = len(characters.tokens)
    else:
        processor = LLGuidanceLogitsProcessor(
            llguidance.grammar_from("lark", grammar()), tokenizer, "numpy"
        )
        proposed = tokenizer.tokenize_str(text)
        assert tokenizer.decode_str(proposed) == text
        eos = tokenizer.eos_token
        vocab_size = tokenizer.vocab_size
    output = []
    accepted = False
    for offset, token in enumerate([*proposed, eos]):
        mask = processor(
            np.array([output], dtype=np.int64),
            np.zeros((1, vocab_size), dtype=np.float32),
        )
        prefix = text[:offset] if tokenizer is None else tokenizer.decode_str(output)
        assert bool(np.isfinite(mask[0, eos])) == _accepts(lalr, prefix), (offset, repr(text))
        if not np.isfinite(mask[0, token]):
            break
        if token == eos:
            accepted = True
        else:
            output.append(token)
    assert _accepts(lalr, text) == _accepts(earley, text) == accepted, repr(text)
    return accepted


def _near_misses(text):
    return [
        text.replace("uL", "ul"),
        text.replace("  add", " add", 1),
        text.replace("  add", "\tadd", 1),
        text[:-1],
        "```text\n" + text + "```\n",
        text.replace("\n\nstep develop", "\nstep develop"),
        text.replace("\n", "\r\n"),
    ]


@pytest.mark.parametrize("name", ["01_overflow", "02_wrong_intent", "03_two_fold"])
def test_examples_near_misses_and_character_mutations(name, parsers):
    text = (EXAMPLES / f"{name}.lab").read_bytes().decode()
    assert _check_document(text, parsers)
    for candidate in _near_misses(text):
        assert not _check_document(candidate, parsers)
    rng = random.Random(20261030)
    for _ in range(12):
        index = rng.randrange(len(text) - 1)
        symbol = rng.choice(' \n\r\t"0129AZaz_-')
        for candidate in (
            text[:index] + text[index + 1 :],
            text[:index] + symbol + text[index:],
            text[:index] + text[index + 1] + text[index] + text[index + 2 :],
        ):
            _check_document(candidate, parsers)


def test_malformed_example(parsers):
    assert not _check_document((EXAMPLES / "00_malformed.lab").read_text(), parsers)


@pytest.mark.parametrize(
    "replacement",
    [
        ("max well volume 200 uL", "max well volume 1000000 uL"),
        ('"Buffer A"', f'"{"x" * 61}"'),
        ("step dilute", f"step {'a' * 33}"),
    ],
)
def test_mask_and_lark_reject_bounded_token_overflow(parsers, replacement):
    text = (EXAMPLES / "03_two_fold.lab").read_text()
    candidate = text.replace(*replacement, 1)

    assert not _check_document(candidate, parsers)
    with pytest.raises(ProtocolSyntaxError):
        parse(candidate)


def test_reversed_range_is_semantic_not_syntactic(parsers):
    text = (EXAMPLES / "03_two_fold.lab").read_text().replace("wells A2-A6", "wells A6-A2")
    assert _check_document(text, parsers)
    assert "RANGE" in [d.code for d in validate(parse(text)).diagnostics]


def test_random_walks_of_real_outlines_mask(parsers):
    lalr, earley = parsers
    completed = 0
    for seed in range(16):
        rng = random.Random(seed)
        characters, processor = _character_processor()
        output = []
        text = ""
        for _ in range(1024):
            mask = processor(
                np.array([output], dtype=np.int64),
                np.zeros((1, len(characters.tokens)), dtype=np.float32),
            )
            allowed = np.flatnonzero(np.isfinite(mask[0])).tolist()
            eos_allowed = characters.eos_token_id in allowed
            assert _accepts(lalr, text) == eos_allowed, (seed, repr(text))
            if eos_allowed:
                assert _accepts(earley, text), (seed, repr(text))
                completed += 1
                break
            # Bias toward delimiters so unbounded names/strings/numbers terminate.
            weights = [12 if characters.characters[t] in '"\n ' else 1 for t in allowed]
            token = rng.choices(allowed, weights=weights)[0]
            output.append(token)
            text += characters.characters[token]
        else:
            assert not _accepts(lalr, text)
            assert not _accepts(earley, text)
    assert completed == 16, f"Only {completed}/16 walks completed within the budget"


def test_tokenizer_only_bpe_crosses_grammar_boundaries(tmp_path, parsers):
    tokenizers = pytest.importorskip(
        "tokenizers", reason="Install the local extra for BPE coverage"
    )
    texts = [
        (EXAMPLES / f"{name}.lab").read_text()
        for name in ("01_overflow", "02_wrong_intent", "03_two_fold")
    ]
    backend = tokenizers.Tokenizer(tokenizers.models.BPE(unk_token="<unk>"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(
        add_prefix_space=False, use_regex=False
    )
    backend.decoder = tokenizers.decoders.ByteLevel()
    backend.train_from_iterator(
        texts,
        tokenizers.trainers.BpeTrainer(
            vocab_size=512,
            min_frequency=2,
            special_tokens=["<unk>", "<eos>"],
            initial_alphabet=tokenizers.pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    path = tmp_path / "tokenizer.json"
    backend.save(str(path))
    tokenizer = llguidance.LLTokenizer(path.read_text(), eos_token=backend.token_to_id("<eos>"))
    pieces = [tokenizer.decode_str([token]) for token in tokenizer.tokenize_str(texts[2])]
    assert any('"\n' in piece or "\n  " in piece for piece in pieces), pieces
    assert len(pieces) < len(texts[2])
    for text in texts:
        assert _check_document(text, parsers, tokenizer)
        for candidate in _near_misses(text):
            assert not _check_document(candidate, parsers, tokenizer)
