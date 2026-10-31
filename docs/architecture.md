# Architecture

Pointer file, not a duplicate. Canonical, detailed architecture — domain model, encoding,
objective math, directive rules, caching strategy, adaptation seams — lives in
[PROJECT.md](../PROJECT.md).

This file exists so the orchestrator loop and `coder`/`docs-writer` sub-agents have a
fixed path (`docs/architecture.md`) to check per the recommended project structure.
When architecture changes (new objective, new encoding, new module boundary), update
PROJECT.md directly — the source of truth — and reflect the corresponding requirement
row in [sdd.md](sdd.md).

## Module boundary summary

```
src/engine/   domain-agnostic NSGA-II problem, solver, directive generation
src/utils/    privacy (HMAC pseudonymization, HIPAA masking), mock data, exporters
src/ui/       Streamlit dashboard (presentation only, no optimization logic)
config/       YAML-driven tuning surface (penalties, topology) — no code changes needed
remotion-video/  standalone TS/React project, independent of src/, own package.json
```

`src/engine/` must remain domain-agnostic (integers/floats in, integers/floats out).
Domain-specific meaning (patients, beds, wards) is confined to `models.py` field names
and `mock_data.py`; the five adaptation seams are documented in PROJECT.md under
"Adapting This Pattern to a New Domain."
