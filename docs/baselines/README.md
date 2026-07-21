# Background001 draft baseline

`background001_current_draft.json` is a deterministic characterization of the
pre-refactor offline pipeline. It is not a target architecture and its quality
numbers are not release thresholds. It preserves observable draft behavior so
later branches can identify intentional changes.

Capture or verify it from the repository root:

```bash
python scripts/capture_background001_baseline.py
python scripts/capture_background001_baseline.py --check docs/baselines/background001_current_draft.json
```

The frozen scenario is `datasets/background001/frozen_replay_v1.json`. Its
fingerprint excludes only the `fixture_fingerprint` field and is checked before
every replay. Network-backed providers are disabled during capture.

The zero token/cost/latency values are explicitly marked as not instrumented.
They document a draft-system observability gap; branches 01, 04, and 10 replace
those placeholders with measured event, kernel, and evaluation data.
