# Provenance note: this lane served Opus 4.8, not Fable 5

Runs in this directory were launched with `--model claude-fable-5`, but this
account lacks gated Fable 5 access, so the API silently served `claude-opus-4-8`
for all 15 runs (no error, nothing in stderr). Verified from each run's
`modelUsage`, whose workhorse model is `claude-opus-4-8`.

Treat this directory as a SECOND Opus 4.8 sample, not as Fable 5. There is no
Fable 5 data in this sweep.
