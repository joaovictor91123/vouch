"""The pluggable-strategy engine lane, with the sandbox as the load-bearing
part. Every escape test here guards the boundary that lets vouch score
untrusted ranking code automatically."""

from pathlib import Path

import pytest

from vouch.strategy import (
    Candidate,
    SandboxProxy,
    apply_ordering,
    load_from_path,
    run_sandboxed,
)

REPO = Path(__file__).resolve().parents[1]
BASELINE = str(REPO / "contrib" / "strategies" / "baseline.py")
EXAMPLE = str(REPO / "contrib" / "strategies" / "example_lexical.py")

CANDS = [
    Candidate("claim", "a", "alpha beta", 0.9),
    Candidate("claim", "b", "gamma delta", 0.5),
    Candidate("page", "c", "beta gamma query", 0.3),
]


def _write(tmp_path: Path, body: str) -> str:
    p = tmp_path / "challenger.py"
    p.write_text(body, encoding="utf-8")
    return str(p)


# --- interface + ordering discipline --------------------------------------


def test_baseline_is_identity() -> None:
    strat = load_from_path(BASELINE)
    assert strat.rank("beta", CANDS, limit=10) == ["a", "b", "c"]


def test_apply_ordering_drops_unknown_and_appends_missing() -> None:
    hits = [(c.kind, c.id, c.summary, c.score) for c in CANDS]
    out = apply_ordering(["c", "invented", "a"], hits)
    # 'invented' dropped, 'b' (unmentioned) appended at the tail.
    assert [h[1] for h in out] == ["c", "a", "b"]


def test_apply_ordering_cannot_grow_the_set() -> None:
    hits = [(c.kind, c.id, c.summary, c.score) for c in CANDS]
    out = apply_ordering(["a", "a", "b", "c", "c"], hits)
    assert sorted(h[1] for h in out) == ["a", "b", "c"]


# --- sandbox happy path ----------------------------------------------------


def test_sandbox_runs_and_matches_in_process() -> None:
    assert run_sandboxed(BASELINE, "beta", CANDS, limit=10) == ["a", "b", "c"]


def test_sandbox_is_deterministic() -> None:
    a = run_sandboxed(EXAMPLE, "beta gamma query", CANDS, limit=10)
    b = run_sandboxed(EXAMPLE, "beta gamma query", CANDS, limit=10)
    assert a == b and a is not None


def test_sandbox_only_returns_known_ids() -> None:
    body = """
from vouch.strategy import Candidate
def rank(query, candidates, *, limit):
    return ["a", "ghost", "c"]  # 'ghost' is not a candidate
"""
    # the child filters to valid ids before returning.
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        path = _write(Path(d), body)
        assert run_sandboxed(path, "q", CANDS, limit=10) == ["a", "c"]


# --- sandbox blocks (the security contract) -------------------------------


def test_sandbox_blocks_network(tmp_path: Path) -> None:
    body = """
from vouch.strategy import Candidate
def rank(query, candidates, *, limit):
    import socket
    socket.socket().connect(("1.1.1.1", 80))
    return [c.id for c in candidates]
"""
    assert run_sandboxed(_write(tmp_path, body), "q", CANDS, limit=10) is None


def test_sandbox_blocks_file_write(tmp_path: Path) -> None:
    target = tmp_path / "pwned"
    body = f"""
from vouch.strategy import Candidate
def rank(query, candidates, *, limit):
    open({str(target)!r}, "w").write("x")
    return [c.id for c in candidates]
"""
    assert run_sandboxed(_write(tmp_path, body), "q", CANDS, limit=10) is None
    assert not target.exists()


def test_sandbox_blocks_subprocess(tmp_path: Path) -> None:
    body = """
from vouch.strategy import Candidate
def rank(query, candidates, *, limit):
    import subprocess
    subprocess.Popen(["/bin/echo", "hi"])
    return [c.id for c in candidates]
"""
    assert run_sandboxed(_write(tmp_path, body), "q", CANDS, limit=10) is None


def test_sandbox_kills_infinite_loop(tmp_path: Path) -> None:
    body = """
from vouch.strategy import Candidate
def rank(query, candidates, *, limit):
    while True:
        pass
"""
    assert (
        run_sandboxed(_write(tmp_path, body), "q", CANDS, limit=10, timeout_s=5)
        is None
    )


def test_audit_hook_survives_module_global_disarm(tmp_path: Path) -> None:
    # the child runs as __main__; reassigning the guard globals must NOT
    # re-enable network/exec, because the hook reads them from closure cells
    # holding the original frozensets.
    body = """
import sys
_m = sys.modules["__main__"]
_m._BLOCKED_EXACT = frozenset()
_m._BLOCKED_PREFIXES = ()
def rank(query, candidates, *, limit):
    import socket
    socket.socket().connect(("1.1.1.1", 80))
    return [c.id for c in candidates]
"""
    assert run_sandboxed(_write(tmp_path, body), "q", CANDS, limit=10) is None


def test_strategy_stdout_does_not_corrupt_result(tmp_path: Path) -> None:
    # a strategy that prints debug output must still have its reordering
    # respected - the result travels on a channel the strategy cannot dirty.
    body = """
import sys
def rank(query, candidates, *, limit):
    print("noisy debug line")
    sys.stdout.write("more noise\\n")
    return list(reversed([c.id for c in candidates]))
"""
    assert run_sandboxed(_write(tmp_path, body), "q", CANDS, limit=10) == [
        "c",
        "b",
        "a",
    ]


def test_sandbox_survives_a_crashing_strategy(tmp_path: Path) -> None:
    body = """
def rank(query, candidates, *, limit):
    raise RuntimeError("boom")
"""
    assert run_sandboxed(_write(tmp_path, body), "q", CANDS, limit=10) is None


def test_proxy_counts_failures_and_returns_empty(tmp_path: Path) -> None:
    body = """
def rank(query, candidates, *, limit):
    raise RuntimeError("boom")
"""
    proxy = SandboxProxy(_write(tmp_path, body))
    assert proxy.rank("q", CANDS, limit=10) == []
    assert proxy.failures == 1


# --- retrieval integration -------------------------------------------------


def test_build_context_pack_strategy_none_is_default(tmp_path: Path) -> None:
    # strategy=None must not change retrieval - exercised in bench parity, but
    # assert the config hook returns None cleanly on a bare KB here.
    from vouch.context import _configured_strategy
    from vouch.storage import KBStore

    store = KBStore.init(tmp_path / "kb")
    assert _configured_strategy(store) is None


@pytest.mark.parametrize("path", [BASELINE, EXAMPLE])
def test_shipped_examples_load(path: str) -> None:
    strat = load_from_path(path)
    assert hasattr(strat, "rank")
    result = strat.rank("beta gamma", CANDS, limit=10)
    assert sorted(result) == ["a", "b", "c"]
