# Deep Agents — Agent Evaluation Framework

> Reference notes on the evaluation framework in `libs/evals/` of the Deep Agents monorepo.

---

## Part 1: Framework Overview

The entire framework lives in `libs/evals/`. It runs Deep Agents end-to-end against real LLMs, captures the full trajectory (tool calls, file mutations, final text), scores correctness + efficiency, and logs everything to LangSmith. There are **118 evals across 8 categories**, plus sandboxed benchmark integration via Harbor.

### 1. Architecture: Two-Tier Assertion Model

The core idea (`tests/evals/utils.py`) is a `TrajectoryScorer` with two assertion tiers:

| Tier | Method | Behavior | Examples |
|---|---|---|---|
| **Success** (correctness) | `.success(...)` | **Hard-fails** the test | `final_text_contains`, `file_equals`, `llm_judge` |
| **Efficiency** (trajectory shape) | `.expect(...)` | **Logged but never fails** | expected `agent_steps`, `tool_call_requests`, `tool_calls` |

```python
scorer = (
    TrajectoryScorer()
    .expect(agent_steps=2, tool_call_requests=1)        # soft targets
    .success(final_text_contains("three", case_insensitive=True))  # must pass
)
```

This separation lets correctness gate CI while efficiency drift is tracked without flapping the build.

### 2. Core Data Structures

- **`AgentStep`** — one `AIMessage` action + its `ToolMessage` observations (1-indexed).
- **`AgentTrajectory`** — list of steps + final `files` dict; `.answer` returns last step text, `.pretty()` renders a readable trace.
- `_trajectory_from_result()` reconstructs the trajectory from a raw `agent.invoke()` result by walking messages.

### 3. Assertion Library

**Success assertions** (hard-fail): `FinalTextContains`, `FinalTextExcludes`, `FinalTextContainsAny`, `FinalTextMinLength`, `FileEquals`, `FileContains`, `FileExcludes`, and `LLMJudge`.

**Efficiency assertions** (logged): `AgentSteps`, `ToolCallRequests`, `MaxToolCallRequests`, `ToolCall` (with optional step pinning).

Each exposes `check(trajectory) -> bool` and `describe_failure(trajectory) -> str`. There's a `_strip_common_zero_width` helper so models inserting invisible Unicode don't break literal substring matches.

### 4. LLM-as-Judge (`tests/evals/llm_judge.py`)

For semantic grading where substring matching is insufficient. It's a `SuccessAssertion` wrapping [openevals](https://github.com/langchain-ai/openevals):
- Each criterion graded **independently**; all must pass.
- Criteria run **concurrently** (`ThreadPoolExecutor`, max 8 workers) with `copy_context()` to preserve tracing context.
- `include_tool_calls=False` → judge sees only agent text; `True` → full trajectory (for "did it actually edit the file" criteria).
- Default judge model: `claude-sonnet-4-6`.

### 5. Entry Point: `run_agent()`

```python
run_agent(agent, query=..., model=model, scorer=..., initial_files=..., thread_id=..., extra_state=...)
```
It builds invoke inputs (seeding `files` and merging middleware state like `rubric`), logs minimal inputs/outputs to LangSmith, reconstructs the trajectory, then runs assertions. There's also an async `run_agent_async`.

### 6. Categories (`deepagents_evals/categories.json` — single source of truth)

8 categories: `file_operations`, `retrieval`, `tool_use`, `memory`, `conversation`, `summarization`, `unit_test`, `langchain/middleware`. The first 6 are `radar_categories` (measure model capability); `unit_test`/middleware are SDK plumbing and excluded from radar charts.

Tests are tagged with `@pytest.mark.eval_category("name")` and `@pytest.mark.eval_tier("baseline"|"hillclimb")`.

### 7. Test Suites (118 evals)

| Suite | Category | Evaluates |
|---|---|---|
| `test_file_operations.py` | file_operations, retrieval | read/write/edit/ls, parallel ops, grep/glob, pagination recovery (13 evals) |
| `test_tool_selection.py` | tool_use | picking right tool from intent with mock tools |
| `test_tool_usage_relational.py` | tool_use | dependent multi-step chaining (user→location→weather) |
| `test_todos.py` | tool_use | todo planning tool |
| `test_external_benchmarks.py` | retrieval, tool_use | **FRAMES** (multi-hop), **Nexus** (nested fn composition), **BFCL v3** (stateful multi-turn) |
| `test_memory.py` / `test_memory_multiturn.py` | memory | AGENTS.md recall, preference persistence, transient filtering |
| `memory_agent_bench/` | memory | **MemoryAgentBench (ICLR 2026)** long-context QA |
| `test_followup_quality.py` | conversation | followup relevance (LLM judge) |
| `tau2_airline/` | conversation | **tau2-bench** airline multi-turn, scored on DB state + communicated info |
| `test_summarization.py` | summarization | summarization middleware triggers, history offload |
| `test_subagents.py`, `test_skills.py`, `test_system_prompt.py`, `test_hitl.py` | unit_test | delegation, SKILL.md, prompt adherence, human-in-the-loop |

External benchmark data is vendored under `tests/evals/data/` and `tau2_airline/data/` (must stay byte-identical to upstream).

### 8. Reporting (`pytest_reporter.py`)

Custom pytest plugin that collects per-test efficiency data and emits a summary:
- **correctness** — fraction passing all success assertions
- **step_ratio** — micro-averaged actual/expected steps
- **tool_call_ratio** — micro-averaged actual/expected tool calls
- **solve_rate** — mean of `expected_steps / duration_s` for passers
- per-category scores, durations, LangSmith experiment links, failure details

Crucially, it **rewrites pytest's exit status to 0** even on eval failures, so CI shell steps don't fail — the CLI instead reads `trials_summary.json`'s `counts.failed.mean` to decide pass/fail.

### 9. CLI (`deepagents-evals`)

The canonical interface (subcommands wrap Makefile/scripts):

| Subcommand | Purpose |
|---|---|
| `run` | single trial |
| `trials` | run N times, aggregate mean/median/stdev/min/max |
| `aggregate` | aggregate prior trial reports / CI artifacts |
| `radar` | radar chart across categories |
| `catalog` / `model-groups` | regenerate or `--check` drift on generated docs |
| `list` | discover categories / tiers / models / evals |

Exit codes are meaningful for automation: `0` success, `1` eval failures, `2` config error / stale generated file, `3` no usable reports. Most subcommands support `--json` and `--dry-run`. Default model via `DEEPAGENTS_EVALS_MODEL`.

### 10. Required Environment & Multi-Trial Signal

The suite **refuses to start** without `LANGSMITH_TRACING=true` + `LANGSMITH_API_KEY` (`conftest.py` aborts early), plus a provider key matching `--model`. Multi-trial runs exist to separate signal from run-to-run variance — two workflows: `evals.yml` (1 trial × many models, for comparison) and `evals_trials.yml` (N trials × 1 model, for stability).

### 11. Harbor Integration (`deepagents_harbor/`)

For sandboxed benchmarks like **Terminal Bench 2.0**. It wraps the agent as a LangGraph project (`deepagents_harbor/langgraph_project/`) runnable across Modal, Daytona, Docker, Runloop, or LangSmith prod sandboxes (Makefile targets). `langsmith.py` handles dataset/experiment/feedback creation; `failure.py` defines `FailureCategory` taxonomy.

### 12. Reproducibility / Drift Guards

Generated docs (`EVAL_CATALOG.md`, `MODEL_GROUPS.md`) have `--check` drift tests in `tests/unit_tests/` that fail CI if stale. `generate_eval_catalog.py` auto-scans test files; radar generation is auto-skipped for narrow category-filtered runs.

### Summary

The framework is a pytest-based, LangSmith-traced, trajectory-scoring harness. You write an eval as a `@pytest.mark.langsmith` test that builds an agent with `create_deep_agent`, calls `run_agent(...)` with a `TrajectoryScorer`, and tags it with a category/tier. Correctness gates the build; efficiency is tracked; LLM-judge handles semantic grading; external academic benchmarks (FRAMES, Nexus, BFCL, tau2, MemoryAgentBench) are integrated; and Harbor extends it to full sandboxed terminal benchmarks.

---

## Part 2: Complete Mental Model

### The One-Sentence Model

> **An eval is a pytest test that runs a real agent against a real LLM, records what the agent *did* (its trajectory), and asserts on that record — with correctness checks that fail the build and efficiency checks that only get logged.**

Everything else is plumbing around that core loop.

### The Central Abstraction: Trajectory → Assertions

The entire framework rests on one transformation:

```
agent.invoke(query)  →  raw messages  →  AgentTrajectory  →  TrajectoryScorer  →  pass/fail + metrics
```

#### Why "trajectory" and not "output"?

A traditional eval checks `output == expected`. But agents don't just produce output — they *act*: they call tools, read/write files, plan, delegate. A correct final answer reached via 14 wasteful tool calls is a *different quality* of result than the same answer in 2 calls.

So the framework reifies the agent's whole run as an **`AgentTrajectory`**:

```
AgentTrajectory
├── steps: [AgentStep, AgentStep, ...]   # the "what it did"
│     └── AgentStep
│           ├── action: AIMessage        # what the agent said/decided (may have tool calls)
│           └── observations: [ToolMessage]  # what the tools returned
└── files: {path: content}               # the world-state it left behind
```

`.answer` = last step's text. `.pretty()` = human-readable trace for debugging and for feeding to the LLM judge.

**Mental hook:** A trajectory is the agent's "flight recorder." You don't just grade the destination — you can grade the whole flight path.

### The Two-Tier Insight (the most important design decision)

This is the conceptual heart. There are two fundamentally different questions you ask about an agent run:

| Question | Tier | What happens if it's wrong |
|---|---|---|
| **"Did it get the right result?"** | `.success()` | **Hard fail** — the test is red, CI gate trips |
| **"Did it get there efficiently?"** | `.expect()` | **Logged only** — never fails the test |

```python
scorer = (
    TrajectoryScorer()
    .expect(agent_steps=2, tool_call_requests=1)   # "I'd expect ~2 steps" — soft
    .success(final_text_contains("Paris"))          # "It MUST say Paris" — hard
)
```

#### Why separate them?

Because **correctness is binary and efficiency is a gradient.**

- If you hard-failed on step count, every model that took 3 steps instead of 2 would go red — your CI would flap constantly and you'd learn nothing.
- If you only logged correctness, regressions would silently ship.

So: **correctness gates, efficiency informs.** You watch efficiency *trend over time* (step_ratio, tool_call_ratio) to catch a prompt change that made the agent 2x more wasteful, without that ever blocking a merge.

**Mental hook:** Success = "did it work?" (a gate). Expect = "how well did it work?" (a gauge).

### The Assertion Vocabulary

Assertions are how you express "what good looks like." Two families:

**Success (hard-fail):**
- Text shape: `final_text_contains`, `final_text_excludes`, `final_text_contains_any`, `final_text_min_length`
- File state: `file_equals`, `file_contains`, `file_excludes`
- Semantic: `llm_judge` — when no substring can capture "is the tone conversational?"

**Efficiency (logged):**
- `agent_steps`, `tool_call_requests`, `max_tool_call_requests`, `tool_call` (can pin a specific tool to a specific step)

**The escape hatch — `llm_judge`:** When correctness is *semantic* not *literal*, you delegate grading to an LLM. Each criterion is graded independently; all must pass. It can grade just the text ("what it said") or the full trajectory ("what it did" — e.g. "the agent actually called `edit_file`").

**Mental hook:** Start with cheap literal assertions. Reach for `llm_judge` only when meaning can't be pinned to a string.

### The Layers (zoom out)

```
┌─────────────────────────────────────────────────────────┐
│  LangSmith — tracing, experiments, cross-model dashboards │  ← observability
├─────────────────────────────────────────────────────────┤
│  deepagents-evals CLI  (run / trials / aggregate / radar) │  ← orchestration
├─────────────────────────────────────────────────────────┤
│  pytest + pytest_reporter  (collection, metrics, summary) │  ← test runner
├─────────────────────────────────────────────────────────┤
│  run_agent() + TrajectoryScorer + assertions             │  ← the core loop
├─────────────────────────────────────────────────────────┤
│  create_deep_agent()  (the system under test)            │  ← the SUT
└─────────────────────────────────────────────────────────┘

         Harbor (parallel track) ──→ sandboxed benchmarks
                                     (Terminal Bench 2.0)
```

Each layer answers a different question:
- **Core loop:** "Did this one run pass?"
- **Reporter:** "What were the aggregate metrics across all evals this run?"
- **CLI/trials:** "Is this number real or just variance?" (run N times)
- **LangSmith/radar:** "How do models compare across capabilities?"

### The Categories Mental Model

Evals are tagged by **capability area** (`@pytest.mark.eval_category`). This isn't just organization — it's how you get a **capability profile** instead of one blurry number.

```
file_operations  ─┐
retrieval         │
tool_use          ├─→ radar chart (model capability fingerprint)
memory            │
conversation      │
summarization    ─┘
unit_test         ─→ SDK plumbing, excluded from radar
langchain/middleware
```

A single "85% correctness" tells you nothing actionable. But "great at tool_use (0.90), weak at memory (0.62)" tells you exactly where to hill-climb. The radar chart makes this a literal shape you can compare across models.

**Tiers** add a second axis: `baseline` (regression gate — must not drop) vs `hillclimb` (progress tracking — trying to improve).

### The Statistical Layer: Signal vs. Noise

LLMs are non-deterministic. A single run showing 0.85 → 0.87 could be a real improvement or pure luck. So:

- `deepagents-evals trials --trials N` runs the same config N times.
- It aggregates **mean / median / stdev / min / max** per metric.
- Now you can say "0.85 ± 0.02" and know whether your prompt change actually moved the needle.

Two workflows encode the two distinct questions:
- `evals.yml`: **1 trial × many models** → "how do these models compare?"
- `evals_trials.yml`: **N trials × 1 model** → "is this model's number stable?"

**Mental hook:** One run is an anecdote. N runs is a measurement.

### A Subtle but Critical Detail: Exit Code Semantics

The `pytest_reporter` plugin **rewrites pytest's exit code to 0 even when evals fail.** This seems wrong until you realize: eval failures are *data*, not *errors*. A CI step running evals shouldn't crash just because a model got 3 wrong — you want the run to complete, aggregate, and report. The **CLI** then reads `trials_summary.json`'s `counts.failed.mean` to decide the real exit code. Automation should key off CLI exit codes (0/1/2/3), never parse human output.

### How You'd Actually Use It — Example Use Cases

#### Use Case 1: "I changed the system prompt. Did I break anything?"
```sh
# Baseline before your change
deepagents-evals trials --model claude-opus-4-7 --trials 3 --report before.json

# ... make your prompt change ...

deepagents-evals trials --model claude-opus-4-7 --trials 3 --report after.json
```
Compare `correctness.mean` and `step_ratio.mean`. If correctness held but step_ratio dropped from 1.3 → 1.1, you made it *more efficient* without breaking it — a clear win. The stdev tells you if the difference is real.

#### Use Case 2: "Which model should we ship?"
```sh
deepagents-evals run --model claude-opus-4-7 --report claude.json
deepagents-evals run --model openai:gpt-5.5 --report gpt.json
deepagents-evals radar --summary combined.json -o radar.png
```
The radar chart gives you a capability fingerprint per model. Maybe Claude wins on memory, GPT wins on tool_use — now you decide based on *your* workload's category mix, not a single leaderboard number.

#### Use Case 3: "I'm adding a new capability — context summarization. How do I prove it works?"
Write a new eval:
```python
@pytest.mark.langsmith
@pytest.mark.eval_category("summarization")
def test_summarization_continues_task(model):
    agent = create_deep_agent(model=model, middleware=[SummarizationMiddleware(...)])
    run_agent(
        agent,
        model=model,
        query="<long conversation that triggers summarization>",
        scorer=(
            TrajectoryScorer()
            .expect(agent_steps=4)
            .success(
                llm_judge("The agent continued the original task after summarizing."),
            )
        ),
    )
```
Regenerate the catalog (`make eval-catalog`), tag it, and it's now part of the regression gate.

#### Use Case 4: "Is the agent efficient, or is it flailing?"
You don't write a *new* test — efficiency is already captured. Run the suite and read `step_ratio` / `tool_call_ratio` from the summary. A ratio of 1.0 means the agent matched expert expectations; 2.5 means it took 2.5x the tool calls a good run needs. Drill into the LangSmith trace (`.pretty()` trajectory) to see *where* it flailed.

#### Use Case 5: "Only the memory evals matter for this PR."
```sh
deepagents-evals run --model openai:gpt-5.5 --eval-category memory --eval-tier baseline
```
Narrow, fast feedback loop. (Radar auto-skips for narrow runs — a 1-axis radar is meaningless.)

#### Use Case 6: "Re-run just the failures from last night's sweep."
```sh
deepagents-evals trials --model openai:gpt-5.5 --trials 1 \
    --retry-failed trial_runs/trials_summary.json
```

#### Use Case 7: "How does the agent do on a real-world terminal benchmark?"
That's Harbor's job — it spins the agent into a LangGraph project and runs it against **Terminal Bench 2.0** inside sandboxes (Docker/Modal/Daytona/etc.), scoring with a `FailureCategory` taxonomy and logging to LangSmith. This is the "in the wild" complement to the controlled in-suite evals.

#### Use Case 8: "Validate against academic benchmarks."
The suite vendors **FRAMES** (multi-hop retrieval), **Nexus** (nested function composition), **BFCL v3** (stateful multi-turn tool calling), **tau2-bench airline** (multi-turn conversation scored on DB state), and **MemoryAgentBench** (long-context memory). You run them like any other eval — they're integrated behind the same `run_agent` / scorer interface.

### The Whole Thing in One Diagram

```
              YOU WRITE                          FRAMEWORK RUNS                    YOU LEARN
  ┌────────────────────────────┐    ┌──────────────────────────────┐   ┌─────────────────────┐
  │ @pytest.mark.langsmith     │    │ run_agent():                 │   │ correctness (gate)  │
  │ @eval_category("memory")   │    │   agent.invoke(query)        │   │ step_ratio (gauge)  │
  │ def test_x(model):         │───▶│   → AgentTrajectory          │──▶│ tool_call_ratio     │
  │   agent = create_deep_     │    │   → run assertions:          │   │ category scores     │
  │           agent(model)     │    │       .success → hard fail   │   │ → radar fingerprint │
  │   run_agent(scorer=        │    │       .expect  → log only    │   │ → trials: mean±stdev│
  │     .expect(...)           │    │   → log to LangSmith         │   │ → CI gate (exit 0/1)│
  │     .success(...))         │    │ pytest_reporter aggregates   │   │ → LangSmith traces  │
  └────────────────────────────┘    └──────────────────────────────┘   └─────────────────────┘
```

### The Five Things to Remember

1. **Grade the trajectory, not just the output** — agents act, so evaluate the actions.
2. **Two tiers:** correctness gates (hard-fail), efficiency informs (logged). This is the load-bearing design choice.
3. **Capability categories** turn one blurry score into an actionable fingerprint.
4. **N trials** convert a noisy anecdote into a real measurement.
5. **Exit codes, not text** drive automation; the reporter deliberately swallows pytest failures so the CLI can decide based on aggregated data.
