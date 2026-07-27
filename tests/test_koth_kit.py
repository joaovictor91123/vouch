"""The koth kit validator — the closed-world allowlist that lets a kit-only
PR auto-merge without a human. Every test here guards a way an untrusted kit
could smuggle something past the gate."""

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "validate_kit",
    Path(__file__).resolve().parents[1] / ".github" / "scripts" / "validate_kit.py",
)
assert _SPEC and _SPEC.loader
validate_kit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(validate_kit)
validate = validate_kit.validate

CHAMPION = (
    Path(__file__).resolve().parents[1]
    / "competition"
    / "kits"
    / "current"
    / "kit.yaml"
).read_text(encoding="utf-8")


def test_champion_kit_validates() -> None:
    assert validate(CHAMPION) == []


def test_empty_kit_is_rejected() -> None:
    # the >1MB contents-api bypass: an oversized file returns empty content,
    # which must NOT validate as "champion defaults".
    assert validate("") != []
    assert validate("   \n  \n") != []


def test_kit_without_retrieval_section_is_rejected() -> None:
    assert validate("review:\n  auto_approve_on_receipt: true\n") != []


def test_command_key_is_rejected() -> None:
    # config can name executables (compile.llm_cmd); a kit must never reach one.
    errors = validate("retrieval:\n  backend: auto\ncompile:\n  llm_cmd: 'sh -c evil'\n")
    assert any("not in allowlist" in e for e in errors)


def test_unknown_retrieval_key_is_rejected() -> None:
    errors = validate("retrieval:\n  backend: auto\n  secret_knob: 1\n")
    assert any("not in allowlist" in e for e in errors)


def test_out_of_bounds_values_are_rejected() -> None:
    assert validate("retrieval:\n  backend: nonsense\n") != []
    assert validate("retrieval:\n  recency:\n    half_life_days: 99999\n") != []
    assert validate("retrieval:\n  rerank:\n    top_k: 0\n") != []
    assert validate("retrieval:\n  pages_first:\n    boost: -1\n") != []


def test_bool_is_not_accepted_as_a_number() -> None:
    # yaml true is an int subclass in python; a numeric knob must refuse it.
    assert validate("retrieval:\n  recency:\n    half_life_days: true\n") != []


def test_oversized_kit_is_rejected() -> None:
    big = "retrieval:\n  backend: auto\n" + ("# pad\n" * 2000)
    assert any("larger than" in e for e in validate(big))
