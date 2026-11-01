# Grammar-constrained generation with semantic feedback

A software demonstration of grammar-constrained generation and semantic
validation for [DSL-mediated agents](https://doi.org/10.5281/zenodo.23587494).

The code in this repository combines grammar masking with DSL parsing and
semantic validation. The simulation and local-model generation loop show how validator
diagnostics can guide bounded retries; a passing DSL document still needs human review.

The example DSL is invented and not laboratory guidance, safety software, or robot control.

👉 [Read the white paper](docs/white-paper.md) for details on how grammar masking
constrains syntax while semantic validation checks encoded rules, with human review reserved
for intent.

## Quickstart

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```sh
uv sync --locked
uv run outlines-lab generate --demo
```

See the [repository usage guide](docs/usage.md) for development checks, local-model
setup, generation, validation, and grammar compatibility testing.

## License

Original project code and documentation are under the
[MIT License](LICENSE); see [NOTICE](NOTICE) for third-party attribution and terms.
