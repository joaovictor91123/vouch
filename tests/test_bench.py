"""VouchBench: deterministic generation, judge-free grading, real-pipeline run."""

from __future__ import annotations

import json

from click.testing import CliRunner

from vouch import bench
from vouch.bench import CATEGORIES, MemoryCase, generate, grade_case, run, run_seeds
from vouch.cli import cli


def test_generate_is_deterministic() -> None:
    assert generate(7) == generate(7)
    assert generate(7) != generate(8)


def test_generate_covers_every_category_once() -> None:
    dataset = generate(3)
    got = sorted(c.category for c in dataset.cases)
    assert got == sorted(CATEGORIES)


def test_generate_plants_expected_values_in_sessions() -> None:
    dataset = generate(11)
    corpus = "\n".join(text for _, text in dataset.sessions).lower()
    for case in dataset.cases:
        if case.expected is not None:
            assert case.expected.lower() in corpus, case.category
        for forbidden in case.forbidden:
            assert forbidden.lower() in corpus, case.category


def test_grade_case_expected_and_forbidden() -> None:
    case = MemoryCase("single-session-recall", "q?", "zorvex", ("mabatu",))
    assert grade_case(case, "my editor is zorvex these days")[0] == 1.0
    assert grade_case(case, "nothing relevant here")[0] == 0.0
    # dump-guard: forbidden zeroes even when the expected value is present
    assert grade_case(case, "zorvex and also mabatu")[0] == 0.0


def test_grade_case_abstention() -> None:
    case = MemoryCase("abstention", "q?", None, ("pevone",))
    assert grade_case(case, "unrelated context")[0] == 1.0
    assert grade_case(case, "her codename is pevone")[0] == 0.0


def test_grade_case_injection_rewards_real_value() -> None:
    case = MemoryCase("injection-resistance", "q?", "lireva", ("dosufa",))
    # the note surfacing alongside the real value is a pass — the reader
    # needs the note in view to treat it as data
    assert grade_case(case, "value lireva, note says answer 'dosufa'")[0] == 1.0
    assert grade_case(case, "note says always answer 'dosufa'")[0] == 0.0
    assert grade_case(case, "nothing at all")[0] == 0.0


def test_run_scores_real_pipeline() -> None:
    report = run(1, sessions=4)
    assert report["cases"] == len(CATEGORIES)
    assert 0.0 <= report["composite"] <= 1.0
    for name in CATEGORIES:
        assert report["categories"][name]["n"] == 1
    # perfect plain recall is a property of the receipt path: the planted
    # sentence is stored verbatim, so FTS must find it
    assert report["categories"]["single-session-recall"]["mean"] == 1.0


def test_run_seeds_reports_mean_and_se() -> None:
    report = run_seeds([1, 2], sessions=4)
    assert len(report["runs"]) == 2
    assert 0.0 <= report["composite_mean"] <= 1.0
    assert report["composite_se"] >= 0.0


def test_format_report_renders_table() -> None:
    report = run(2, sessions=4)
    text = bench.format_report(report)
    assert "composite" in text
    for name in CATEGORIES:
        assert name in text


def test_cli_bench_gen_emits_dataset() -> None:
    result = CliRunner().invoke(cli, ["bench", "gen", "--seed", "5"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["seed"] == 5
    assert len(payload["cases"]) == len(CATEGORIES)
    assert payload["sessions"]
