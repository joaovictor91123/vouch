"""Per-artifact effectiveness — issue #426."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from vouch import health, lifecycle, proposals, retrieval_events, sessions
from vouch.eval import effectiveness as eff
from vouch.storage import KBStore


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> KBStore:
    kb = KBStore.init(tmp_path)
    monkeypatch.chdir(kb.root)
    kb.config_path.write_text(
        "retrieval:\n  events:\n    enabled: true\n", encoding="utf-8"
    )
    return kb


def _claim(store: KBStore, text: str) -> str:
    src = store.put_source(text.encode("utf-8") + b" source bytes")
    pr = proposals.propose_claim(
        store, text=text, evidence=[src.id], proposed_by="agent"
    )
    return proposals.approve(store, pr.id, approved_by="reviewer").id


def _session(store: KBStore, *, surfaced: str, outcome: str, task: str) -> None:
    """One session that surfaced `surfaced` and ended good or bad."""
    sess = sessions.session_start(store, agent="agent", task=task)
    retrieval_events.log_event(
        store, query=task, backend="fts5", limit=5, budget_chars=2000,
        items=[{"type": "claim", "id": surfaced, "score": 1.0}],
    )
    if outcome == "good":
        lifecycle.confirm(store, claim_id=surfaced, actor="reviewer")
    else:
        lifecycle.archive(store, claim_id=surfaced, actor="reviewer")
    sessions.session_end(store, sess.id)


def _by_id(result: dict) -> dict[str, dict]:
    return {a["id"]: a for a in result["artifacts"]}


# --- the wilson interval --------------------------------------------------


def test_wilson_interval_brackets_the_point_estimate() -> None:
    low, high = eff.wilson_interval(8, 10)
    assert low < 0.8 < high
    assert low >= 0.0 and high <= 1.0


def test_wilson_interval_stays_inside_the_unit_range_at_the_edges() -> None:
    """The reason for Wilson over the normal approximation: p=0 and p=1."""
    assert eff.wilson_interval(0, 5)[0] == 0.0
    assert eff.wilson_interval(5, 5)[1] == 1.0
    assert eff.wilson_interval(0, 5)[1] < 1.0
    assert eff.wilson_interval(5, 5)[0] > 0.0


def test_wilson_interval_narrows_as_the_sample_grows() -> None:
    small = eff.wilson_interval(3, 4)
    large = eff.wilson_interval(75, 100)
    assert (large[1] - large[0]) < (small[1] - small[0])


def test_wilson_interval_of_no_samples_is_maximally_uncertain() -> None:
    assert eff.wilson_interval(0, 0) == (0.0, 1.0)


# --- event classification -------------------------------------------------


@pytest.mark.parametrize("event", ["claim.confirm", "proposal.claim.approve"])
def test_good_events_classify_good(event: str) -> None:
    assert eff.classify_event(event) == "good"


@pytest.mark.parametrize(
    "event",
    [
        "claim.contradict", "claim.supersede", "claim.archive", "claim.redact",
        "proposal.claim.reject", "proposal.page.reject",
    ],
)
def test_bad_events_classify_bad(event: str) -> None:
    assert eff.classify_event(event) == "bad"


@pytest.mark.parametrize("event", ["source.add", "kb.init", "session.start"])
def test_neutral_events_carry_no_verdict(event: str) -> None:
    assert eff.classify_event(event) is None


# --- session outcome ------------------------------------------------------


def test_mixed_session_is_dropped_rather_than_guessed(store: KBStore) -> None:
    """A session that confirmed one claim and archived another says nothing."""
    kept = _claim(store, "auth uses jwt tokens")
    dropped = _claim(store, "auth uses saml everywhere")
    sess = sessions.session_start(store, agent="agent", task="mixed")
    retrieval_events.log_event(
        store, query="auth", backend="fts5", limit=5, budget_chars=2000,
        items=[{"type": "claim", "id": kept, "score": 1.0}],
    )
    lifecycle.confirm(store, claim_id=kept, actor="reviewer")
    lifecycle.archive(store, claim_id=dropped, actor="reviewer")
    ended = sessions.session_end(store, sess.id)

    assert eff.session_outcome(store, ended) is None
    assert eff.compute(store)["sessions_scored"] == 0


def test_open_session_is_not_scored(store: KBStore) -> None:
    claim = _claim(store, "auth uses jwt tokens")
    sessions.session_start(store, agent="agent", task="still running")
    lifecycle.confirm(store, claim_id=claim, actor="reviewer")
    assert eff.compute(store)["sessions_scored"] == 0


# --- the insufficient-sample path ----------------------------------------


def test_insufficient_samples_withhold_a_verdict(store: KBStore) -> None:
    """Under min_samples nothing is called useful, however clean the signal."""
    claim = _claim(store, "auth uses jwt tokens")
    for i in range(2):
        _session(store, surfaced=claim, outcome="good", task=f"good-{i}")

    result = eff.compute(store, min_samples=5)
    entry = _by_id(result)[claim]
    assert entry["samples"] == 2
    assert entry["rate"] == 1.0
    assert entry["verdict"] == eff.VERDICT_INSUFFICIENT


def test_wide_interval_reports_unverified_not_a_guess(store: KBStore) -> None:
    """Sample meets the floor but the interval still straddles the baseline."""
    good = _claim(store, "auth uses jwt tokens")
    bad = _claim(store, "auth uses saml everywhere")
    for i in range(3):
        _session(store, surfaced=good, outcome="good", task=f"g{i}")
        _session(store, surfaced=bad, outcome="bad", task=f"b{i}")

    result = eff.compute(store, min_samples=3)
    assert result["baseline"] == 0.5
    assert {a["verdict"] for a in result["artifacts"]} == {eff.VERDICT_UNVERIFIED}


# --- the clear-signal path ------------------------------------------------


def test_clear_signal_earns_useful_and_harmful(store: KBStore) -> None:
    good = _claim(store, "auth uses jwt tokens")
    bad = _claim(store, "auth uses saml everywhere")
    for i in range(8):
        _session(store, surfaced=good, outcome="good", task=f"g{i}")
        _session(store, surfaced=bad, outcome="bad", task=f"b{i}")

    result = eff.compute(store, min_samples=3)
    by_id = _by_id(result)
    assert by_id[good]["verdict"] == eff.VERDICT_USEFUL
    assert by_id[good]["lift"] > 0
    assert by_id[good]["ci_low"] > result["baseline"]
    assert by_id[bad]["verdict"] == eff.VERDICT_HARMFUL
    assert by_id[bad]["lift"] < 0
    assert by_id[bad]["ci_high"] < result["baseline"]


def test_artifacts_are_ranked_worst_first(store: KBStore) -> None:
    """The reviewer's queue is the dead weight, not the winners."""
    good = _claim(store, "auth uses jwt tokens")
    bad = _claim(store, "auth uses saml everywhere")
    for i in range(4):
        _session(store, surfaced=good, outcome="good", task=f"g{i}")
        _session(store, surfaced=bad, outcome="bad", task=f"b{i}")

    lifts = [a["lift"] for a in eff.compute(store)["artifacts"]]
    assert lifts == sorted(lifts)


def test_window_excludes_older_sessions(store: KBStore) -> None:
    from vouch import metrics as metrics_mod

    claim = _claim(store, "auth uses jwt tokens")
    _session(store, surfaced=claim, outcome="good", task="recent")

    assert eff.compute(store)["sessions_scored"] == 1
    future = metrics_mod.parse_since("2099-01-01")
    assert eff.compute(store, since=future)["sessions_scored"] == 0


def test_empty_kb_reports_no_artifacts_and_a_zero_baseline(store: KBStore) -> None:
    result = eff.compute(store)
    assert result == {
        "baseline": 0.0, "sessions_scored": 0, "min_samples": 5, "artifacts": [],
    }


# --- the read-only invariant ----------------------------------------------


def test_compute_writes_nothing(store: KBStore) -> None:
    """No artifact, no audit event, no proposal — measurement only."""
    claim = _claim(store, "auth uses jwt tokens")
    for i in range(3):
        _session(store, surfaced=claim, outcome="good", task=f"g{i}")
    health.rebuild_index(store)

    from vouch import audit as audit_mod

    before_claims = {c.id for c in store.list_claims()}
    before_events = len(list(audit_mod.read_events(store.kb_dir)))
    before_pending = len(store.list_proposals())
    before_status = {c.id: c.status for c in store.list_claims()}

    eff.compute(store)

    assert {c.id for c in store.list_claims()} == before_claims
    assert len(list(audit_mod.read_events(store.kb_dir))) == before_events
    assert len(store.list_proposals()) == before_pending
    assert {c.id: c.status for c in store.list_claims()} == before_status


# --- report + surfaces ----------------------------------------------------


def test_format_report_renders_a_table(store: KBStore) -> None:
    claim = _claim(store, "auth uses jwt tokens")
    for i in range(3):
        _session(store, surfaced=claim, outcome="good", task=f"g{i}")

    text = eff.format_report(eff.compute(store, min_samples=2))
    assert "sessions scored: 3" in text
    assert "verdict" in text
    assert claim in text


def test_format_report_says_so_when_there_is_nothing(store: KBStore) -> None:
    assert "no artifact was surfaced" in eff.format_report(eff.compute(store))


def test_jsonl_surface_serves_effectiveness(
    store: KBStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vouch import jsonl_server

    claim = _claim(store, "auth uses jwt tokens")
    _session(store, surfaced=claim, outcome="good", task="g0")
    monkeypatch.setattr(jsonl_server, "_store", lambda: store)

    result = jsonl_server.HANDLERS["kb.effectiveness"]({"window": "all", "min_samples": 1})
    assert result["min_samples"] == 1
    assert claim in _by_id(result)


def test_mcp_surface_serves_effectiveness(
    store: KBStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vouch import server

    claim = _claim(store, "auth uses jwt tokens")
    _session(store, surfaced=claim, outcome="good", task="g0")
    monkeypatch.setattr(server, "_store", lambda: store)

    result = server.kb_effectiveness(window="all", min_samples=1)
    assert claim in _by_id(result)


def test_effectiveness_is_registered_in_capabilities() -> None:
    from vouch.capabilities import METHODS

    assert "kb.effectiveness" in METHODS


def test_cli_text_output(store: KBStore) -> None:
    from vouch.cli import cli

    claim = _claim(store, "auth uses jwt tokens")
    for i in range(3):
        _session(store, surfaced=claim, outcome="good", task=f"g{i}")

    res = CliRunner().invoke(
        cli, ["eval", "effectiveness", "--window", "all", "--min-samples", "2"]
    )
    assert res.exit_code == 0, res.output
    assert "sessions scored: 3" in res.output
    assert eff.VERDICT_USEFUL in res.output or eff.VERDICT_UNVERIFIED in res.output


def test_cli_json_output_is_machine_readable(store: KBStore) -> None:
    from vouch.cli import cli

    claim = _claim(store, "auth uses jwt tokens")
    _session(store, surfaced=claim, outcome="good", task="g0")

    res = CliRunner().invoke(
        cli, ["eval", "effectiveness", "--format", "json", "--window", "all"]
    )
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert payload["sessions_scored"] == 1
    assert payload["artifacts"][0]["id"] == claim


# --- resilience to a malformed / uninformative log ------------------------


def test_session_outcome_of_an_unfinished_session_is_none(store: KBStore) -> None:
    """Called directly: no end timestamp means no window to classify."""
    open_session = sessions.session_start(store, agent="agent", task="open")
    assert open_session.ended_at is None
    assert eff.session_outcome(store, open_session) is None


def test_session_with_only_neutral_events_has_no_outcome(store: KBStore) -> None:
    """source.add and session.* carry no verdict, so the session is dropped."""
    sess = sessions.session_start(store, agent="agent", task="neutral")
    store.put_source(b"just registering a source, no review happened")
    ended = sessions.session_end(store, sess.id)

    assert eff.session_outcome(store, ended) is None
    assert eff.compute(store)["sessions_scored"] == 0


def test_malformed_retrieval_events_are_skipped_not_fatal() -> None:
    """The event log tolerates junk lines, so the reader must too."""
    from datetime import UTC, datetime

    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = datetime(2026, 1, 2, tzinfo=UTC)
    events = [
        {"items": [{"type": "claim", "id": "no-ts"}]},          # ts missing
        {"ts": 12345, "items": [{"type": "claim", "id": "int-ts"}]},  # ts not a str
        {"ts": "not-a-timestamp", "items": [{"type": "claim", "id": "bad-ts"}]},
        {"ts": "2025-06-01T00:00:00+00:00",                     # outside the window
         "items": [{"type": "claim", "id": "too-early"}]},
        {"ts": "2026-01-01T12:00:00+00:00",
         "items": ["not-a-dict", {"type": "claim", "id": ""},
                   {"type": "claim", "id": "kept"}]},
    ]
    assert eff._surfaced_in(events, start, end) == {("claim", "kept")}
