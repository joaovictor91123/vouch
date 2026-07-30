"""VouchBench: deterministic generation, judge-free grading, real-pipeline run."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from vouch import bench, health, lifecycle
from vouch.bench import (
    CATEGORIES,
    DEFAULT_SESSIONS,
    MemoryCase,
    generate,
    grade_case,
    run,
    run_seeds,
)
from vouch.cli import cli
from vouch.context import build_context_pack
from vouch.extract import ingest_source
from vouch.storage import KBStore


def _mini_kb(tmp_path: Path, *sentences: str) -> KBStore:
    store = KBStore.init(tmp_path / "kb")
    store.config_path.write_text(
        "review:\n  auto_approve_on_receipt: true\n", encoding="utf-8"
    )
    for sentence in sentences:
        ingest_source(store, sentence.encode("utf-8"), proposed_by="bench-test")
    health.rebuild_index(store)
    return store


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


def test_categories_include_verifiability_axes() -> None:
    assert "citation-correctness" in CATEGORIES
    assert "receipt-coverage" in CATEGORIES
    assert "supersede-hygiene" in CATEGORIES


def test_grade_citation_correctness_requires_verifying_receipt(
    tmp_path: Path,
) -> None:
    store = _mini_kb(
        tmp_path, "for the record, my favorite editor is zorvex right now."
    )
    case = MemoryCase("citation-correctness", "what is my favorite editor?", "zorvex")
    pack = dict(
        build_context_pack(store, query=case.question, limit=10, max_chars=2000)
    )
    assert bench.grade_citation_correctness(store, case, pack)[0] == 1.0
    # the value surfacing WITHOUT a receipt that verifies is worth nothing —
    # a claim item citing a bare/unknown id has no mechanical backing
    unbacked = {
        "items": [
            {
                "id": "x",
                "type": "claim",
                "summary": "my favorite editor is zorvex",
                "citations": ["deadbeef"],
            }
        ]
    }
    score, reason = bench.grade_citation_correctness(store, case, unbacked)
    assert score == 0.0
    assert reason is not None


def test_grade_receipt_coverage_full_on_receipt_backed_pack(
    tmp_path: Path,
) -> None:
    store = _mini_kb(tmp_path, "the project codename is mulopi now.")
    case = MemoryCase("receipt-coverage", "what is the project codename?", "mulopi")
    pack = dict(
        build_context_pack(store, query=case.question, limit=10, max_chars=2000)
    )
    assert bench.grade_receipt_coverage(store, case, pack)[0] == 1.0
    unbacked = {
        "items": [
            {
                "id": "x",
                "type": "claim",
                "summary": "the codename is mulopi",
                "citations": ["deadbeef"],
            }
        ]
    }
    assert bench.grade_receipt_coverage(store, case, unbacked)[0] == 0.0


def test_grade_supersede_hygiene_rewards_lifecycle(tmp_path: Path) -> None:
    store = _mini_kb(
        tmp_path,
        "for the record, my deploy day is monday right now.",
        "heads up, the deploy day changed to friday this week.",
    )
    case = MemoryCase(
        "supersede-hygiene", "which day do we deploy?", "friday", ("monday",)
    )
    # stock pipeline leaves the stale value live — the lever this category creates
    score, reason = bench.grade_supersede_hygiene(store, case)
    assert score == 0.0
    assert reason is not None
    old = next(c for c in store.list_claims() if "monday" in c.text)
    new = next(c for c in store.list_claims() if "friday" in c.text)
    lifecycle.supersede(store, old_claim_id=old.id, new_claim_id=new.id, actor="t")
    assert bench.grade_supersede_hygiene(store, case)[0] == 1.0


def test_run_verifiability_stock_scores() -> None:
    report = run(1, sessions=4)
    # guard categories: the receipt path makes these perfect on the stock
    # engine; a change that surfaces unbacked content loses points here
    assert report["categories"]["receipt-coverage"]["mean"] == 1.0
    assert report["categories"]["citation-correctness"]["mean"] == 1.0
    # lever category: nothing in the no-model ingest path supersedes yet
    assert report["categories"]["supersede-hygiene"]["mean"] == 0.0


def test_pack_text_strips_highlight_markers() -> None:
    # retrieval wraps query-matched terms in guillemets; grading must see
    # the text, not the markup, or substring checks fail exactly on the
    # query-relevant claims (and highlighted forbidden values sneak past)
    pack = {"items": [{"summary": "my limit is 340 «requests» «per» «minute»"}]}
    text = bench._pack_text(pack)
    assert "340 requests per minute" in text
    case = MemoryCase("single-session-recall", "q", "340 requests per minute")
    assert grade_case(case, text)[0] == 1.0


def test_paired_verdict_floor_gatekeeps_zero_se() -> None:
    # identical per-seed diffs collapse the SE; the floor still gates
    verdict = bench.paired_verdict([0.5, 0.5, 0.5], [0.505, 0.505, 0.505])
    assert verdict["se"] == 0.0
    assert verdict["band"] == bench.DETHRONE_FLOOR
    assert not verdict["dethroned"]
    clear = bench.paired_verdict([0.5, 0.5, 0.5], [0.51, 0.51, 0.51])
    assert clear["dethroned"]


def test_paired_verdict_requires_same_seed_count() -> None:
    with pytest.raises(ValueError):
        bench.paired_verdict([0.5, 0.5], [0.5])


def test_run_seeds_threads_strategy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[object] = []

    def fake_run(seed: int, **kwargs: object) -> dict:
        seen.append(kwargs.get("strategy"))
        return {
            "seed": seed,
            "composite": 0.5,
            "categories": {name: {"mean": 0.5} for name in CATEGORIES},
        }

    monkeypatch.setattr(bench, "run", fake_run)
    sentinel = object()
    bench.run_seeds([1, 2], strategy=sentinel)
    assert seen == [sentinel, sentinel]


def test_cli_bench_run_against_requires_strategy(tmp_path: Path) -> None:
    champion = tmp_path / "champ.py"
    champion.write_text("def rank(query, candidates, *, limit):\n    return []\n")
    result = CliRunner().invoke(cli, ["bench", "run", "--against", str(champion)])
    assert result.exit_code != 0
    assert "--against requires --strategy" in result.output


# --- derivation categories (#617) -----------------------------------------

_DERIVATION_CATEGORIES = (
    "passive-consolidation",
    "multi-hop-relational",
    "temporal-depth",
    "aggregation",
)


def test_categories_include_derivation_axes() -> None:
    for name in _DERIVATION_CATEGORIES:
        assert name in CATEGORIES


def test_derivation_cases_state_no_answer_only_parts() -> None:
    """The graded fact must be derivable and never stated.

    Each derivation case carries `required` parts instead of an `expected`
    answer — if it carried an expected string the generator would have had to
    write that string into a session, which is the degeneration into
    single-session recall the category exists to avoid.
    """
    cases = [c for c in generate(11).cases if c.category in _DERIVATION_CATEGORIES]
    assert len(cases) == len(_DERIVATION_CATEGORIES)
    for case in cases:
        assert case.expected is None, case.category
        assert len(case.required) >= 3, case.category
        assert len(set(case.required)) == len(case.required), case.category


def test_derivation_parts_are_never_answerable_from_one_session() -> None:
    """Every required part is in the corpus, and no session holds them all.

    Deliberately not "one part per session": ``sub_spread`` only keeps the
    parts in distinct sessions while the session count allows, and a five-part
    case under ``sessions=4`` shares. What the category actually needs is the
    weaker property asserted here — that no single session answers the
    question on its own.
    """
    # sessions=3 crowds five parts into fewer sessions than parts, which is
    # where sub_spread stops being able to keep them distinct — the invariant
    # has to survive there, not just at the roomy default.
    for sessions in (DEFAULT_SESSIONS, 4, 3):
        dataset = generate(4, sessions=sessions)
        bodies = [text for _title, text in dataset.sessions]
        corpus = "\n".join(bodies)
        for case in dataset.cases:
            if case.category not in _DERIVATION_CATEGORIES:
                continue
            for part in case.required:
                assert part in corpus, (sessions, case.category, part)
            assert not any(
                all(part in body for part in case.required) for body in bodies
            ), f"{case.category} is answerable from one session at {sessions=}"


def test_derivation_answer_key_is_a_pure_function_of_the_seed() -> None:
    for seed in (1, 9, 23):
        first = [c for c in generate(seed).cases if c.required]
        second = [c for c in generate(seed).cases if c.required]
        assert first == second
    assert [c.required for c in generate(1).cases if c.required] != [
        c.required for c in generate(2).cases if c.required
    ]


def test_existing_categories_are_unchanged_by_derivation_cases() -> None:
    """The ten original categories must generate identically.

    The derivation block draws from a derived rng and its own pools precisely
    so recorded per-category scores stay comparable across this change. If a
    future edit advances the main rng before category 10, this fails.
    """
    original = [c for c in generate(1).cases if not c.required]
    assert [c.category for c in original] == [
        "single-session-recall",
        "multi-session",
        "knowledge-update",
        "point-in-time",
        "decoy-discrimination",
        "injection-resistance",
        "abstention",
        "citation-correctness",
        "receipt-coverage",
        "supersede-hygiene",
    ]
    # pinned against the pre-#617 generator for seed 1 — these are the values
    # the ten categories produced before the derivation block existed, so a
    # change here means the main rng stream moved and recorded scores shifted
    assert original[0].expected == "zedo-2"
    assert original[2].expected == "tenare"
    assert original[2].forbidden == ("babubo",)
    assert original[4].expected == "410 requests per minute"
    assert original[9].expected == "9:05"


def test_grade_required_all_parts_present() -> None:
    case = MemoryCase("aggregation", "list all", None, required=("alpha", "beta"))
    assert grade_case(case, "alpha and beta both surfaced") == (1.0, None)


def test_grade_required_reports_the_shortfall() -> None:
    case = MemoryCase(
        "passive-consolidation", "what is on it", None,
        required=("alpha", "beta", "gamma"),
    )
    score, reason = grade_case(case, "only alpha here")
    assert score == 0.0
    assert reason is not None
    assert "1/3 parts" in reason
    assert "'beta'" in reason and "'gamma'" in reason


def test_grade_required_is_case_insensitive() -> None:
    case = MemoryCase("temporal-depth", "history", None, required=("Alpha", "BETA"))
    assert grade_case(case, "alpha then beta") == (1.0, None)


def test_forbidden_still_zeroes_a_derivation_case() -> None:
    """The dump-guard outranks derivation: a leak zeroes before parts count."""
    case = MemoryCase(
        "aggregation", "list all", None,
        forbidden=("leaked",), required=("alpha",),
    )
    score, reason = grade_case(case, "alpha and leaked")
    assert score == 0.0
    assert reason is not None
    assert "forbidden" in reason


def test_run_scores_every_derivation_category() -> None:
    """The real pipeline produces a graded number for each new axis."""
    report = run(1, sessions=4)
    for name in _DERIVATION_CATEGORIES:
        entry = report["categories"][name]
        assert entry["n"] == 1
        assert entry["mean"] is not None
        assert 0.0 <= entry["mean"] <= 1.0
