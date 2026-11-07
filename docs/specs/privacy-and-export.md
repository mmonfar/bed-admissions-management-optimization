# Spec: privacy and export

Synthetic data only. Not a medical device.

- `pseudonymize(id)` returns `PID-` plus 12 hex characters (keyed HMAC-SHA256). Stable in one process; the key is random per process.
- `mask_string` replaces SSN, dates, ZIP, e-mail, phone and name patterns with tags.
- `sanitize_dict` pseudonymises the named fields and masks other strings.
- `to_json` gives `summary` and `actions`; `to_markdown` gives a table, or "No directives generated." for an empty plan.
