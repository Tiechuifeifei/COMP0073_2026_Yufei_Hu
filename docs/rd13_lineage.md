# RD13 lineage (dissertation §§4.3 / 4.6)

Canonical name in this package: **RD13 = RD13_v2** (alias-cleaned reconstruction).

## Path selection (IC surpass chain)

Development loops used to motivate the RD13 factor set follow the IC-surpass
path documented in Chapter 4. Use **Table 4.2 in the report (static; see
`results/manifest.csv`)**.

## Configuration / implementation audit (§4.6)

Issues called out in the dissertation (test-window feedback, cache reuse,
Loop-2 correction injection, `$factor` template, Amihud scaling, aliases)
are narrative/audit items. Machine extract for `tab:config-issues` was
**not** auto-generated; treat the dissertation table as authoritative until
a verified CSV is added.

## Related result packs

- RD8 matched rerun summaries: `results/ch4/`
- Unified ABCD / 2×2 supplements: `results/appE/`, `results/ch4/tab_twobytwo.csv`
- Downstream D-series: `results/ch5/tab_dseries.csv`

Phase-2 exploratory search trees and abandoned reruns are separate from the
frozen RD13_v2 downstream replication artefacts.
