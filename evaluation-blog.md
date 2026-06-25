# Agent Evaluation — Unified Framework

> A collated evaluation framework synthesizing three perspectives:
> - A four-class library API (`AccuracyEval`, `ReliabilityEval`, `PerformanceEval`, `AgentAsJudgeEval`)
> - A pytest + tracing-backed trajectory-scoring harness with a two-tier assertion model
> - A first-principles mental model (outcome → trajectory → step; capability / reliability / safety)
>
> Read top-to-bottom this is a complete, opinionated playbook for evaluating AI agents.

---

## Part 1 — The Core Mental Model

### Why agents need their own evaluation lens

Evaluating an agent is fundamentally different from evaluating a model. A **model** maps input → output in one shot. An **agent** runs a *loop*: it perceives, reasons, calls tools, observes results, and repeats until it decides it's done. That loop changes what "good" means — and it means you must grade **what the agent *did*, not just what it produced**.

### The three nested layers

```
┌─────────────────────────────────────────────────┐
│  OUTCOME      "Did it achieve the goal?"          │  ← what the user cares about
│  ┌───────────────────────────────────────────┐   │
│  │  TRAJECTORY  "Was the path sane & efficient?"│  ← how it got there
│  │  ┌─────────────────────────────────────┐   │   │
│  │  │  STEP   "Was each action correct?"   │   │   │  ← the atomic decisions
│  │  └─────────────────────────────────────┘   │   │
│  └───────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

The key insight: **an agent can get the right answer for the wrong reasons (lucky), or do everything right and still fail (unlucky environment).** A good framework separates these so you know *why* a score is what it is.

### The capability / reliability / safety split

- **Capability** — Can it do the task at all? (the *ceiling*)
- **Reliability** — Does it do it *consistently* across runs and variations? (the *variance*)
- **Alignment / Safety** — Does it avoid harmful, deceptive, or out-of-scope actions even under pressure? (the *floor*)

Most teams over-index on capability and ignore reliability, which is where agents actually break in production (non-determinism, tool flakiness, cascading errors).

### Three ways to frame "is my agent good?"

| Lens | Decomposition | Core abstraction |
|---|---|---|
| First-principles | outcome → trajectory → step; capability / reliability / safety | dimensions × methods |
| Library API | accuracy / reliability / performance / custom-judge | four eval classes |
| Trajectory harness | success (correctness) vs. expect (efficiency) | the **trajectory** + two-tier scorer |

**The four-class split** — divide "is my agent good?" into three orthogonal questions plus a flexible fourth:

```
                  Is the agent GOOD?
                          │
   ┌──────────────┬───────┴───────┬──────────────────┐
   ▼              ▼               ▼                  ▼
ACCURACY      RELIABILITY     PERFORMANCE       AGENT-AS-JUDGE
"Right        "Did it use the "Is it fast &     "Does it meet
 answer?"      right tools?"   memory-light?"    custom criteria?"
   │              │               │                  │
LLM judge     Deterministic   Timer + memory    LLM judge w/
1–10 score    tool-call check  profiling        rubric + threshold
(needs gold   (no LLM, no      (no LLM, no       (no gold answer
 answer)       gold answer)     gold answer)      needed)
```
- **Accuracy & Agent-as-Judge** use an LLM to *judge quality* (subjective, scored).
- **Reliability** is a *deterministic* assertion on the tool-call trace (objective, no LLM).
- **Performance** *measures* runtime + memory over N iterations (no judgment at all).

**The one-sentence trajectory model** — *An eval is a test that runs a real agent against a real LLM, records what the agent did (its **trajectory**), and asserts on that record — with correctness checks that fail the build and efficiency checks that only get logged.*

---

## Part 2 — Evaluation Dimensions (the "what to measure")

| Dimension | Question it answers | Example metrics |
|---|---|---|
| **Task success** | Did it complete the goal? | Pass rate, exact-match, rubric score, % subgoals met |
| **Trajectory quality** | Was the path efficient/valid? | # steps, # tool calls, redundant actions, backtracks |
| **Tool use correctness** | Right tool, right args, right time? | Tool-selection accuracy, arg validity, hallucinated-tool rate |
| **Efficiency / cost** | What did it cost? | Tokens, $/task, wall-clock latency, # API calls |
| **Robustness** | Does it survive perturbation? | Success under noisy input, tool failures, adversarial prompts |
| **Reliability** | Is it consistent? | pass@k vs pass^k, variance across seeds |
| **Safety & alignment** | Does it stay in bounds? | Harmful-action rate, refusal correctness, data-exfil attempts |
| **Recovery** | Can it self-correct? | Error-recovery rate after an injected failure |
| **Grounding / honesty** | Does it hallucinate or fabricate? | Citation accuracy, unsupported-claim rate |

**Critical distinction — `pass@k` vs `pass^k`** (echoed by multi-trial runs and `num_iterations`):
- `pass@k` = succeeds *at least once* in k tries → measures the *capability ceiling*.
- `pass^k` = succeeds in *all* k tries → measures *reliability*. **Production cares about this one.**

---

## Part 3 — The Methods (the "how to measure")

### a) Programmatic / verifiable checks
Best when there's ground truth. Did the file get created? Does the code compile and pass tests? Is the API call's effect present in the DB? **Cheap, deterministic, trustworthy — use these whenever possible.**

- **`ReliabilityEval`** is the canonical packaged form: a fully deterministic check on the tool-call trace, no LLM, designed for CI/pytest.
- **Success assertions** include literal text/file checks: `final_text_contains`, `file_equals`, `file_contains`, etc.

### b) LLM-as-judge
For open-ended outputs (was the support reply empathetic *and* correct?). Use a strong model with a **rubric** and few-shot examples. Cheap and scalable but noisy — must be calibrated against human labels and watched for bias (length bias, self-preference, position bias).

- Two packaged flavors:
  - `AccuracyEval` — LLM judge that **requires a gold/expected answer**, scores 1–10.
  - `AgentAsJudgeEval` — LLM judge with **no gold answer**, scoring against a `criteria` rubric (`numeric` 1–10 + threshold, or `binary` pass/fail).
- A trajectory-harness `llm_judge` (wrapping [openevals](https://github.com/langchain-ai/openevals)): each criterion graded independently, all must pass; criteria run concurrently; can see text-only or the full trajectory.

### c) Human evaluation
Gold standard for nuanced quality, safety edge cases, and **calibrating your judges**. Expensive — reserve for a sampled subset and for validating automated graders (target judge–human agreement ≳ 0.8 before trusting a judge).

### d) Trajectory / process evaluation
Inspect the *sequence* of actions, not just the end state. Did it call the refund tool before verifying the order? Did it loop? This catches "right answer, dangerous path" cases. This is reified as the **`AgentTrajectory`** abstraction (below).

### e) Simulation / sandbox environments
Run the agent against a fake-but-faithful environment (mock APIs, seeded DB, simulated user) so you can test destructive actions safely and reproducibly. A sandbox layer extends this to full sandboxed terminal benchmarks (Terminal Bench 2.0 across Docker/Modal/Daytona/etc.).

### f) Performance / cost profiling
**`PerformanceEval`** measures (doesn't judge): wraps any callable, runs it `num_iterations` times with optional `warmup_runs`, and profiles wall-clock time + memory via `tracemalloc`. It reports `avg/min/max/std_dev` plus **`median` and `p95`** — the tail-latency focus is deliberate, since averages hide tail latency.

---

## Part 4 — The Central Abstraction: Trajectory → Assertions

The trajectory-scoring approach rests on one transformation:

```
agent.invoke(query)  →  raw messages  →  AgentTrajectory  →  Scorer  →  pass/fail + metrics
```

**Why "trajectory" and not "output"?** A traditional eval checks `output == expected`. But agents *act*: they call tools, read/write files, plan, delegate. A correct final answer reached via 14 wasteful tool calls is a *different quality* of result than the same answer in 2 calls.

```
AgentTrajectory
├── steps: [AgentStep, AgentStep, ...]       # the "what it did"
│     └── AgentStep
│           ├── action: AIMessage            # what the agent decided (may have tool calls)
│           └── observations: [ToolMessage]  # what the tools returned
└── files: {path: content}                   # the world-state it left behind
```

`.answer` = last step's text. `.pretty()` = human-readable trace for debugging and for feeding to the LLM judge. **Mental hook:** a trajectory is the agent's "flight recorder" — you grade the whole flight path, not just the destination.

### The Two-Tier Insight (the load-bearing design choice)

| Question | Tier | If wrong |
|---|---|---|
| **"Did it get the right result?"** | `.success()` | **Hard fail** — CI gate trips |
| **"Did it get there efficiently?"** | `.expect()` | **Logged only** — never fails the test |

```python
scorer = (
    TrajectoryScorer()
    .expect(agent_steps=2, tool_call_requests=1)   # soft target — a gauge
    .success(final_text_contains("Paris"))          # must pass — a gate
)
```

**Because correctness is binary and efficiency is a gradient.** If you hard-failed on step count, every model taking 3 steps instead of 2 would go red and CI would flap. If you only logged correctness, regressions would silently ship. So: **correctness gates, efficiency informs** — you watch `step_ratio` / `tool_call_ratio` *trend over time* without ever blocking a merge.

This maps cleanly onto the others: `ReliabilityEval.assert_passed()` is a hard gate; `PerformanceEval` is a logged gauge. The first-principles view phrases the same idea as **hard gates on safety-critical behaviors, trends/averages elsewhere.**

---

## Part 5 — The Framework Loop (mental workflow)

```
1. DEFINE     → success criteria + failure modes you care about (be specific)
2. BUILD      → a dataset of tasks (golden set) with known-good outcomes
3. INSTRUMENT → log full trajectories: prompts, tool calls, args, results, tokens
4. GRADE      → programmatic where possible, LLM-judge for open-ended, human to calibrate
5. AGGREGATE  → scorecard across dimensions, segmented by task type & difficulty
6. ANALYZE    → cluster failures by root cause (tool error vs reasoning vs hallucination)
7. ITERATE    → fix highest-impact failure class, re-run, track regression
```

The dataset (step 2) is the most undervalued piece. Start with **20–50 hand-curated tasks** that cover the *failure modes you've actually seen*, not random happy-path tasks. Grow it by mining production failures.

### The layered architecture in practice

```
┌─────────────────────────────────────────────────────────┐
│  Tracing — experiments, cross-model dashboards            │  ← observability
├─────────────────────────────────────────────────────────┤
│  evals CLI  (run / trials / aggregate / radar)            │  ← orchestration
├─────────────────────────────────────────────────────────┤
│  pytest + reporter  (collection, metrics, summary)        │  ← test runner
├─────────────────────────────────────────────────────────┤
│  run_agent() + TrajectoryScorer + assertions             │  ← the core loop
├─────────────────────────────────────────────────────────┤
│  create_agent()  (the system under test)                 │  ← the SUT
└─────────────────────────────────────────────────────────┘
         Sandbox (parallel track) ──→ sandboxed benchmarks
```

Each layer answers a different question — core loop: "did this one run pass?"; reporter: "what were aggregate metrics this run?"; CLI/trials: "is this number real or just variance?"; tracing/radar: "how do models compare across capabilities?"

### Signal vs. noise: run it N times

LLMs are non-deterministic; a single run is an anecdote, N runs is a measurement.
- The CLI's `evals trials --trials N` aggregates **mean / median / stdev / min / max** per metric, so you can report "0.85 ± 0.02".
- Every eval class takes `num_iterations` for statistical stability; results carry `avg_score`, `min/max`, `std_dev`.

Two distinct workflows encode two questions: **1 trial × many models** ("how do models compare?") vs. **N trials × 1 model** ("is this number stable?").

### Capability fingerprint via categories

Evals are tagged by **capability area** (e.g. `file_operations`, `retrieval`, `tool_use`, `memory`, `conversation`, `summarization`). A single "85% correctness" tells you nothing actionable; "great at tool_use (0.90), weak at memory (0.62)" tells you exactly where to hill-climb. A **radar chart** makes this a literal shape comparable across models. **Tiers** add a second axis: `baseline` (regression gate — must not drop) vs `hillclimb` (progress tracking).

### Exit-code semantics for automation

Eval failures are *data*, not *errors*. The reporter deliberately rewrites pytest's exit code to 0 even when evals fail, so a CI step completes, aggregates, and reports rather than crashing. The **CLI** then reads aggregated counts to decide the real exit code (`0` success, `1` eval failures, `2` config error, `3` no usable reports). **Automation keys off CLI exit codes, never parsed human output.**

---

## Part 6 — Worked Example: a Customer-Support Agent

**The agent:** handles support tickets. Tools: `lookup_order(id)`, `issue_refund(order_id, amount)`, `search_kb(query)`, `escalate_to_human(reason)`.

### Step 1 — Define success & failure modes
- **Success:** resolves the ticket correctly, follows refund policy, tone is helpful, escalates when appropriate.
- **Failure modes that matter:** (1) refunds an order it never verified; (2) hallucinates a policy not in the KB; (3) fails to escalate an angry/edge case; (4) refunds the wrong amount; (5) loops or burns excessive tokens.

### Step 2 — Build the golden dataset

| ID | Ticket | Expected outcome | Tests failure mode |
|---|---|---|---|
| T1 | "Item arrived broken, order #A123, want refund" | Verify order → issue full refund → confirm | Happy path |
| T2 | "Refund me $500 for order #B999" (order is $50) | Refund only $50, not $500 | #4 wrong amount |
| T3 | "I'm furious, 3rd time, I want a manager" | Escalate to human | #3 missing escalation |
| T4 | "Is there a 90-day return window?" (KB says 30) | State 30 days, don't fabricate | #2 hallucination |
| T5 | "Refund order #C404" (doesn't exist) | Look up, find none, clarify — never refund | #1 unverified refund |

### Step 3 — Instrument (log every trajectory)
```json
{
  "task_id": "T2",
  "steps": [
    {"thought": "verify order first", "tool": "lookup_order", "args": {"id": "B999"}, "result": {"amount": 50}},
    {"thought": "user asked $500 but order is $50, refund actual", "tool": "issue_refund", "args": {"order_id": "B999", "amount": 50}}
  ],
  "final_response": "I've refunded the $50 you paid for order #B999.",
  "tokens": 1840, "latency_ms": 3200, "tool_calls": 2
}
```

### Step 4 — Grade (mix of methods)

**Programmatic / process check** — (deterministic `ReliabilityEval`-style tool-call assertion + success-tier `tool_call` step pinning):
```python
def grade_T2(trajectory):
    refunds = [s for s in trajectory.steps if s.tool == "issue_refund"]
    assert len(refunds) == 1, "should refund exactly once"
    assert refunds[0].args["amount"] == 50, "must refund actual amount, not requested"
    # process check: must verify BEFORE refunding
    assert index_of(trajectory, "lookup_order") < index_of(trajectory, "issue_refund"), \
        "must verify order before refunding"
```

**LLM-as-judge** — (rubric-based `AgentAsJudgeEval` / `llm_judge`):
```
RUBRIC (score 1–5 each):
- Accuracy: Did the response match the verified facts?
- Policy adherence: Followed refund/return policy?
- Tone: Empathetic and professional?
- Escalation: Escalated iff the situation warranted it?
Return JSON: {accuracy, policy, tone, escalation, rationale}
```
Calibrate against ~30 human-labeled examples; if judge–human agreement < ~0.8, fix the rubric before trusting it.

### Step 5 — Aggregate into a scorecard

| Metric | Value | Target |
|---|---|---|
| Task success (pass@1) | 84% | ≥90% |
| Reliability (pass^5) | 61% | ≥80% |
| Unverified-refund rate | 2% | **0%** (hard gate) |
| Hallucinated-policy rate | 5% | <1% |
| Correct-escalation rate | 78% | ≥95% |
| Avg tokens/task | 1.9k | <2.5k |
| Avg cost/task | $0.04 | — |

### Step 6 — Analyze failures
Cluster them: suppose 70% of T4-type failures are the agent answering policy questions *from memory* instead of calling `search_kb`. **Root cause = tool-use gap, not reasoning.** Fix: strengthen the system prompt / add a guard that policy questions must hit the KB.

### Step 7 — Iterate & guard against regression
Re-run the full suite after the fix. Wire the **hard gates** (unverified-refund rate = 0%) into CI so any future change that reintroduces the financial-risk behavior fails the build.

---

## Part 7 — Reliability & Safety: the part most teams skip

- **Run each task k times** (different seeds/temperature). Report `pass^k`, not one lucky run. (Via `num_iterations` or `--trials N`.)
- **Inject failures:** make a tool return an error/timeout and check the agent recovers rather than hallucinating success.
- **Adversarial / red-team set:** prompt injections, data-exfiltration attempts, jailbreaks. Track harmful-action rate as a **floor metric with a hard gate**.
- **Drift monitoring:** keep running the eval suite on a schedule in production — providers update models under you, and tools change. (Via live post-hooks or scheduled CI workflows.)

---

## Part 8 — Concrete API / Tooling Cheat-Sheets

### The four eval classes (library API)

```python
# 1. AccuracyEval — LLM judge vs. a GOLD answer (1–10)
from eval.accuracy import AccuracyEval
evaluation = AccuracyEval(
    model=OpenAIChat(id="o4-mini"),                 # the JUDGE
    agent=Agent(model=OpenAIChat(id="gpt-4o"), tools=[CalculatorTools()]),
    input="What is 10*5 then to the power of 2?",
    expected_output="2500",
    num_iterations=3,                                # run 3x for a stable score
)
result = evaluation.run(print_results=True)
assert result.avg_score >= 8

# 2. ReliabilityEval — deterministic tool-call assertion (no LLM, for CI)
from eval.reliability import ReliabilityEval
response = agent.run("What is 10 * 5?")
ReliabilityEval(
    agent_response=response,
    expected_tool_calls=["multiply"],
    expected_tool_call_arguments={"multiply": {"a": 10, "b": 5}},  # partial-match
    allow_additional_tool_calls=False,
).run().assert_passed()                              # raises if not PASSED

# 3. PerformanceEval — measures runtime + memory (no judgment)
from eval.performance import PerformanceEval
PerformanceEval(func=run_agent, num_iterations=5, warmup_runs=1).run(
    print_results=True, print_summary=True)         # reports avg/min/max/median/p95

# 4. AgentAsJudgeEval — LLM judge vs. a RUBRIC (no gold answer)
from eval.agent_as_judge import AgentAsJudgeEval
AgentAsJudgeEval(
    criteria="Explanation should be clear, beginner-friendly, avoid jargon",
    scoring_strategy="numeric", threshold=7, on_fail=on_fail, db=sync_db,
).run(input="Explain what an API is", output=str(resp.content))
```

| Layer | Eval class | Runs in | Catches |
|-------|-----------|---------|---------|
| Correctness | `AccuracyEval` | nightly / pre-release | wrong answers |
| Behavior | `ReliabilityEval` | **CI / pytest** | wrong/missing tool calls |
| Quality | `AgentAsJudgeEval` | CI + live post-hook | tone, clarity, rubric drift |
| Cost | `PerformanceEval` | nightly benchmark | latency & memory regressions |

Shared conventions: sync `.run()` + async `.arun()`; `num_iterations` for stability; `db=...` to persist & query runs; `print_results`/`print_summary` rich tables; `file_path_to_save_results` to disk. All four work on both `Agent` and `Team`.

### Write an eval as a test (harness API)

```python
@pytest.mark.eval_category("summarization")
@pytest.mark.eval_tier("baseline")
def test_summarization_continues_task(model):
    agent = create_agent(model=model, middleware=[SummarizationMiddleware(...)])
    run_agent(
        agent, model=model,
        query="<long conversation that triggers summarization>",
        scorer=(
            TrajectoryScorer()
            .expect(agent_steps=4)                                       # gauge
            .success(llm_judge("Agent continued the original task after summarizing."))  # gate
        ),
    )
```

- **Success assertions (hard-fail):** `final_text_contains`, `final_text_excludes`, `final_text_contains_any`, `final_text_min_length`, `file_equals`, `file_contains`, `file_excludes`, `llm_judge`.
- **Efficiency assertions (logged):** `agent_steps`, `tool_call_requests`, `max_tool_call_requests`, `tool_call` (with optional step pinning).
- **Reporting metrics:** correctness, `step_ratio`, `tool_call_ratio`, `solve_rate`, per-category scores.
- **Integrated academic benchmarks:** FRAMES (multi-hop retrieval), Nexus (nested fn composition), BFCL v3 (stateful multi-turn tool calling), tau2-bench airline (multi-turn scored on DB state), MemoryAgentBench (long-context memory). A sandbox layer adds Terminal Bench 2.0 in sandboxes.
- **Requires** tracing enabled + an API key + a provider key matching `--model`.

**CLI cheat-sheet:**

| Subcommand | Purpose |
|---|---|
| `run` | single trial |
| `trials` | run N times, aggregate mean/median/stdev/min/max |
| `aggregate` | aggregate prior trial reports / CI artifacts |
| `radar` | radar chart across categories |
| `list` | discover categories / tiers / models / evals |

Common workflows: *"did my prompt change break anything?"* → `trials` before/after, compare `correctness.mean` + `step_ratio.mean`; *"which model to ship?"* → `run` per model + `radar`; *"only memory matters this PR"* → `run --eval-category memory --eval-tier baseline`; *"re-run failures"* → `trials --retry-failed trials_summary.json`.

---

## Part 9 — Anti-patterns to avoid

1. **Only grading the final answer.** You miss dangerous-path successes and can't diagnose failures. Always log and grade trajectories.
2. **Single-run evals.** Hides variance; reliability is invisible. Use `pass^k` / N trials.
3. **Uncalibrated LLM judges.** A judge you haven't checked against humans is a confident liar. Validate agreement first.
4. **Change-detector / snapshot tests.** Tests that freeze a current value break on every benign update and add no behavioral coverage. Assert *invariants and relationships*, not snapshots.
5. **Happy-path-only datasets.** Real value is in edge cases and known failure modes.
6. **No cost/latency tracking.** A "correct" agent that costs $2 and takes 90s per task may be unshippable.
7. **Averaging away the dangerous tail.** "98% safe" can mean 2% catastrophic. Use hard gates for safety-critical behaviors, not averages.

---

## Part 10 — Quick-Reference Scorecard Template

```
AGENT EVAL SCORECARD
├── Capability
│   ├── pass@1 ............ task success on first try
│   └── subgoal coverage .. % of required steps achieved
├── Reliability
│   ├── pass^k ............ all-k-runs success
│   └── variance .......... stddev across seeds
├── Efficiency
│   ├── tokens/task, $/task
│   └── steps/task, latency
├── Tool use
│   ├── selection accuracy, arg validity
│   └── hallucinated-tool rate
├── Process (trajectory)
│   ├── order-constraint violations (e.g. verify-before-act)
│   └── redundant/backtrack steps
└── Safety  ← HARD GATES, not averages
    ├── harmful-action rate = 0
    ├── unauthorized-action rate = 0
    └── recovery-after-failure rate
```

---

## TL;DR — The Unified Playbook

1. **Grade the trajectory, not just the output** — agents *act*, so evaluate the actions (the `AgentTrajectory` abstraction; the three layers).
2. **Two tiers / two metric kinds:** correctness *gates* (hard-fail), efficiency *informs* (logged). Reserve **hard gates** for safety-critical behaviors (`.success`/`.expect`; `assert_passed` vs perf).
3. **Pick the cheapest sufficient method:** programmatic > calibrated LLM-judge > human spot-check; profile cost/latency separately (the four eval classes map onto this directly).
4. **Run it N times** to separate signal from noise — report `pass^k` and `mean ± stdev`, not a lucky single run.
5. **Turn one blurry score into a capability fingerprint** via categories + radar, and drive automation off exit codes, not text.
6. **Build the dataset from real failure modes**, then run the tight **define → instrument → grade → analyze → iterate** loop with regression gates wired into CI.
