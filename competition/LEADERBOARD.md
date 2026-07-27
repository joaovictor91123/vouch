# koth ladder — throne history

every row is a dethrone: a kit-only PR that beat the reigning kit by the
margin band and auto-merged on the ladder branch. the merged PR is the
authoritative record; this file is the human-readable ledger, appended by
the maintainer when the monthly season closes and payouts are computed.

the daily throne is provisional (public seeds — see docs/koth-ladder.md);
payout rank is settled by the monthly sealed commit-reveal run.

| # | champion | PR | dethroned on | scored mean | margin over prior |
|---|----------|----|--------------|-------------|-------------------|
| 0 | baseline kit (repo defaults) | — | 2026-07-28 | 0.52 ± 0.03 (seeds 1–6) | — |

payouts follow the season shares in docs/vouchbench-seasons.md
(65/14/10/7/4). days-on-throne accrue between dethrones; the monthly
commit-reveal scored run settles the season standings.
