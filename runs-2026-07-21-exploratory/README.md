# Exploratory multi-model pilot expansion (2026-07-21)

**Not confirmatory. Not preregistered.** These runs were collected before the
grading rubric was frozen or the reference ranges reviewed. Their purpose is to
preview the model comparison, give the rubric diverse material to be piloted
against, and stress-test the harness ahead of the real sweep. They must not be
reported as preregistered confirmatory data.

- Prompts: the five in `runs-2026-04-27/prompts.tsv`, verbatim.
- Repetitions: three per prompt.
- Models: `claude-fable-5`, `claude-opus-4-8`, `claude-sonnet-5`, one subdirectory each.
- Harness: copied verbatim from `runs-2026-04-27/`. The environment (connected
  MCP servers, visible skills) was not restricted, matching the April pilot
  rather than the controlled conditions specified for the confirmatory sweep.
- Each run's `transcript.jsonl` records the session initialization event as the
  provenance record.
