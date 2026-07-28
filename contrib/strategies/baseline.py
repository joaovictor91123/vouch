"""The reigning engine-lane champion: trust the backend's fused order.

This is the strategy every submission must beat. It returns the candidates in
exactly the order retrieval handed them over - i.e. it does nothing - so a
challenger only dethrones it by making vouch's benchmark score genuinely
higher, not by accident.

A strategy is real ranking code. It receives the query and an over-fetched
candidate pool (data only - no KB, no disk, no network) and returns the ids in
the order the reader should see them; the top ``limit`` survive the cut.
Ordering is authoritative but bounded: ids you invent are ignored, and any
candidate you drop is appended at the tail before the cut - so you can
de-prioritise a candidate out of the pack, but never fabricate a result or
shrink the pack.
"""

from vouch.strategy import Candidate


def rank(query: str, candidates: list[Candidate], *, limit: int) -> list[str]:
    return [c.id for c in candidates]
