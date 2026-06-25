# Agno Evaluation Framework

## Mental Model

Agno splits "is my agent good?" into **three orthogonal questions**, plus a flexible fourth. Each is a separate eval class living in `libs/agno/agno/eval/`. They don't overlap — you typically run all of them.

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

The key mental distinction:

- **Accuracy & Agent-as-Judge** = use an LLM to *judge quality* (subjective, scored).
- **Reliability** = *deterministic* assertion on the agent's tool-call trace (objective, no LLM).
- **Performance** = *measures* runtime + memory over N iterations (no judgment at all).

All four work on both `Agent` and `Team`, support sync (`.run()`) and async (`.arun()`), can persist results to any DB (`db=...`), and can save to a file (`file_path_to_save_results`).

---

## 1. AccuracyEval — "Did it get the right answer?"

An **LLM-as-judge that requires a gold/expected answer**. It runs your agent, then a separate evaluator agent (default `OpenAIChat("o4-mini")`) scores the output 1–10 against `expected_output`.

**Use case:** Math/tool agent where there's a correct answer.

```python
from agno.eval.accuracy import AccuracyEval
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.calculator import CalculatorTools

evaluation = AccuracyEval(
    model=OpenAIChat(id="o4-mini"),          # the JUDGE
    agent=Agent(model=OpenAIChat(id="gpt-4o"), tools=[CalculatorTools()]),
    input="What is 10*5 then to the power of 2? do it step by step",
    expected_output="2500",
    additional_guidelines="Output should include the steps and final answer.",
    num_iterations=3,                         # run 3x for a stable score
)
result = evaluation.run(print_results=True)
assert result.avg_score >= 8
```

**Output** (`AccuracyResult`): per-run `score`/`reason`, plus aggregate `avg_score`, `min/max`, `std_dev_score`. Multiple iterations matter because LLM judging is noisy — std dev tells you how flaky the agent is.

---

## 2. ReliabilityEval — "Did it call the right tools, with the right args?"

**Fully deterministic, no LLM.** It inspects the run's message trace and asserts which tools were called (and optionally their arguments). This is the one you put in CI/pytest.

**Use case:** Ensure a "10!" prompt actually triggers the `factorial` tool, not a hallucinated mental-math answer.

```python
from agno.eval.reliability import ReliabilityEval

response = agent.run("What is 10 * 5?")
evaluation = ReliabilityEval(
    agent_response=response,
    expected_tool_calls=["multiply"],
    expected_tool_call_arguments={"multiply": {"a": 10, "b": 5}},  # partial-match args
    allow_additional_tool_calls=False,        # strict: no extra tools allowed
)
result = evaluation.run(print_results=True)
result.assert_passed()                        # raises if eval_status != "PASSED"
```

**Output** (`ReliabilityResult`): `passed/failed_tool_calls`, `missing_tool_calls`, `additional_tool_calls`, `passed/failed_argument_checks`. For teams it flattens member responses too. `assert_passed()` makes it a drop-in test assertion.

Notable design: arguments are **partial-matched** (you only specify the keys you care about), and you can assert a tool was called multiple times by passing a list of arg-dicts.

---

## 3. PerformanceEval — "Is it fast and memory-efficient?"

**Measures, doesn't judge.** Wraps any callable, runs it `num_iterations` times (with optional `warmup_runs`), and profiles wall-clock time + memory via `tracemalloc`.

**Use case:** Catch latency/memory regressions, e.g. "does adding memory + reasoning blow up runtime?"

```python
from agno.eval.performance import PerformanceEval

def run_agent():
    agent = Agent(model=OpenAIChat(id="gpt-5.2"), system_message="Be concise.")
    return agent.run("What is the capital of France?")

perf = PerformanceEval(func=run_agent, num_iterations=5, warmup_runs=1)
perf.run(print_results=True, print_summary=True)
```

**Output** (`PerformanceResult`): for both runtime (sec) and memory (MiB) it reports `avg`, `min`, `max`, `std_dev`, **`median`, and `p95`**. The p95/median focus is deliberate — averages hide tail latency. The cookbook has variants isolating instantiation cost vs. response cost vs. memory/storage overhead.

---

## 4. AgentAsJudgeEval — "Does it meet my custom criteria?"

**LLM-as-judge with NO gold answer** — instead you give a `criteria` rubric. This is for open-ended quality (tone, clarity, safety) where there's no single right answer. Two scoring strategies:

- `"numeric"` → 1–10 score + `threshold` (e.g. fail if < 7)
- `"binary"` → pass/fail

**Use case:** "Is this explanation beginner-friendly?" — judged against a rubric, with an `on_fail` callback for alerting.

```python
from agno.eval.agent_as_judge import AgentAsJudgeEval, AgentAsJudgeEvaluation

def on_fail(ev: AgentAsJudgeEvaluation):
    print(f"FAILED {ev.score}/10: {ev.reason}")

evaluation = AgentAsJudgeEval(
    criteria="Explanation should be clear, beginner-friendly, avoid jargon",
    scoring_strategy="numeric",
    threshold=7,
    on_fail=on_fail,
    db=sync_db,                               # persists every eval run
)
resp = agent.run("Explain what an API is")
evaluation.run(input="Explain what an API is", output=str(resp.content),
               print_results=True, print_summary=True)
```

**Output** (`AgentAsJudgeResult`): per-run pass/fail + reason, plus `pass_rate`, and (numeric mode only) `avg/min/max/std_dev` scores. It can also run as a **post-hook** on the agent itself (see `agent_as_judge_post_hook.py`) so production runs are scored live.

---

## How they fit together (the workflow)

`BaseEval` (`eval/base.py`) defines a `pre_check`/`post_check` (sync + async) contract — the hook-style evals (agent-as-judge) implement it so they can run inline on real traffic. The standalone evals are run explicitly.

A realistic eval suite for one agent:

| Layer | Eval | Runs in | Catches |
|-------|------|---------|---------|
| Correctness | `AccuracyEval` | nightly / pre-release | wrong answers |
| Behavior | `ReliabilityEval` | **CI / pytest** | wrong/missing tool calls |
| Quality | `AgentAsJudgeEval` | CI + live post-hook | tone, clarity, rubric drift |
| Cost | `PerformanceEval` | nightly benchmark | latency & memory regressions |

**Shared conventions across all four:**

- Sync `.run()` + async `.arun()`.
- `num_iterations` for statistical stability (LLM judging is non-deterministic).
- `db=PostgresDb(...)` / `AsyncSqliteDb(...)` to persist runs → queryable via `db.get_eval_runs()` and viewable in AgentOS.
- `print_results` / `print_summary` render rich tables; `file_path_to_save_results` dumps to disk.
- Telemetry on by default (minimal, `AGNO_DEBUG`/env-controlled).

The cookbook folder `cookbook/09_evals/` has runnable examples for every variant (batch, custom evaluator, guidelines, team, post-hook, multi-user).

---

## Source Reference

| Eval | Source file | Cookbook |
|------|-------------|----------|
| Base contract | `libs/agno/agno/eval/base.py` | — |
| Accuracy | `libs/agno/agno/eval/accuracy.py` | `cookbook/09_evals/accuracy/` |
| Reliability | `libs/agno/agno/eval/reliability.py` | `cookbook/09_evals/reliability/` |
| Performance | `libs/agno/agno/eval/performance.py` | `cookbook/09_evals/performance/` |
| Agent-as-Judge | `libs/agno/agno/eval/agent_as_judge.py` | `cookbook/09_evals/agent_as_judge/` |
