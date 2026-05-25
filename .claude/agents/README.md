# Subagent definitions

These are [Claude Code subagents](https://docs.claude.com/en/docs/claude-code/sub-agents)
for running the DRC pipeline. Each maps to a stage of the PRD and *drives the
real modules* under `src/drc/` — they operate the code, they don't reinvent it.

| Agent | PRD role | Stage it runs |
|-------|----------|---------------|
| `orchestrator` | Master | Dispatches the others in order, gates on acceptance tests |
| `data-prep` | A | download → parse → filters → QA audit |
| `dose-corpora` | B | build the 20 dose-level corpora + sanity checks |
| `tokenizer` | C | train the shared 16k BPE tokenizer |
| `training-sweep` | D | pilot, then the 60-run sweep |
| `evaluation` | E | SLOR evaluation + n-gram baseline |
| `statistical-analysis` | F | Hill fits, model comparison, clustering, decision |
| `figure-maker` | G | the five paper figures |
| `paper-writer` | H | fill the manuscript from results (post-review) |

The orchestrator stops for human review after `figure-maker`, before
`paper-writer`, exactly as the PRD specifies. Every agent's acceptance test is
the one written in `docs/PRD.md` §10.
