# Agent Procedural Memory

A production-grade library for mining, retrieving, consolidating, generalizing, and incrementally refining agent procedural memory skills from OpenTelemetry-shaped execution traces, with optional LLM augmentation.

The library accepts OpenTelemetry-shaped agent traces, mines successful and failed workflow patterns, computes Bayesian confidence scores per step, optionally enriches outputs via LLM, emits a portable `SKILL.md`-style memory, and supports:

- **Embedding-based skill retrieval** from a skill library (Jaccard fallback when no embeddings)
- **Cross-task generalization** via frequent sub-procedure extraction
- **Online learning** — incrementally refine skills from new traces without re-mining
- **Three-condition evaluation** (baseline, few-shot, procedural skill) with bootstrap CIs

## Research Lineage

The design mirrors recent work on agent memory, skill learning, and cognitive architectures:

- [ACT-R](https://act-r.psy.cmu.edu/) (Anderson, 1993) — Production rule strength and base-level activation as models for procedural memory confidence. Inspires our Bayesian confidence scoring and temporal decay.
- [CoALA](https://arxiv.org/abs/2309.02427) (Sumers et al., 2024) — Cognitive Architectures for Language Agents. Identifies procedural memory as a core agent component; distinguishes grounded, abstracted, and composed procedures.
- [Reflexion](https://arxiv.org/abs/2303.11366) — Turns task feedback and failures into verbal memory for future trials.
- [Voyager](https://arxiv.org/abs/2305.16291) — Maintains a growing library of executable skills and uses curriculum-style exploration.
- [ExpeL](https://arxiv.org/abs/2308.10144) — Generalizes lessons from successful and failed trajectories.
- [ExpeTrans](https://arxiv.org/abs/2505.23191) — Studies experiential transfer, reinforcing the idea that textual experience can improve future task solving.
- [Toolformer](https://arxiv.org/abs/2302.04761) (Schick et al., 2023) — The tool-augmented LLM pattern: LLMs enhance but do not replace deterministic algorithms. Motivates our dual-mode architecture.
- [LLM-as-Judge](https://arxiv.org/abs/2306.05685) (Zheng et al., 2023) — Validates strong LLMs as evaluation proxies. Grounds our LLM-as-judge scoring option.
- [Frequent Sequential Patterns](https://dl.acm.org/doi/10.5555/645920.672836) (Agrawal & Srikant, 1995) — Mining Sequential Patterns. Grounds our contiguous n-gram sub-procedure extraction.
- [Online Bayesian Updating](https://www.wiley.com/en-us/Bayesian+Theory-p-9780471924166) (Bernardo & Smith, 2000) — Beta-Binomial conjugate model for incremental learning from binary outcomes.
- [Exgentic Agent LLM Traces v2](https://huggingface.co/datasets/Exgentic/agent-llm-traces-v2/tree/main) and its dataset note, [What Agents Do](https://huggingface.co/blog/michalshmu/exgentic-agent-llm-traces-v2), motivate using OpenTelemetry GenAI traces as the raw material for skill generation.

## Install

```bash
python3 -m pip install -e .
```

No runtime dependencies are required. Optional extras:

```bash
# Load public Hugging Face dataset
python3 -m pip install -e ".[public-data]"

# Enable LLM-augmented mining (Anthropic SDK)
python3 -m pip install -e ".[llm]"

# Enable embedding-based retrieval (OpenAI SDK)
python3 -m pip install -e ".[embeddings]"

# Install everything
python3 -m pip install -e ".[all]"
```

## Quick Start

### Deterministic mode (no LLM required)

```python
from agent_procedural_memory import (
    load_otel_jsonl,
    ProceduralMemoryMiner,
    EvaluationSuite,
)

traces = load_otel_jsonl("data/public_trace_fixtures/support_refund_otel.jsonl")
skill = ProceduralMemoryMiner().mine_skill(traces, skill_name="refund_workflow")
print(skill.to_markdown())
```

### LLM-augmented mode

```python
from agent_procedural_memory import (
    load_otel_jsonl,
    ProceduralMemoryMiner,
    LLMConfig,
)

class MyLLMBackend:
    def complete(self, prompt: str, *, temperature: float = 0.4, max_tokens: int = 1024) -> str:
        # Wrap any LLM provider here (Anthropic, OpenAI, local model, etc.)
        import anthropic
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text

traces = load_otel_jsonl("data/public_trace_fixtures/support_refund_otel.jsonl")
llm_config = LLMConfig(backend=MyLLMBackend(), temperature=0.4, max_retries=2)
miner = ProceduralMemoryMiner(llm_config=llm_config)
skill = miner.mine_skill(traces, skill_name="refund_workflow")
print(skill.to_markdown())
```

### Evaluation with bootstrap confidence intervals

```python
from agent_procedural_memory import EvaluationSuite
from agent_procedural_memory.toy_agent import RefundWorkflowAgent, build_refund_eval_cases

suite = EvaluationSuite(
    build_refund_eval_cases(),
    RefundWorkflowAgent(),
    bootstrap_samples=1000,
    seed=42,
)
report = suite.compare(skill, few_shot_traces=few_shot)
print(report.to_text())
```

### Skill consolidation

```python
from agent_procedural_memory import consolidate_skills, ConsolidationConfig

# Merge overlapping skills from different mining runs
consolidated = consolidate_skills([skill_a, skill_b, skill_c])
```

### Skill retrieval

```python
from agent_procedural_memory import SkillIndex

# Index skills (uses Jaccard token similarity by default)
index = SkillIndex()
index.add_many([skill_a, skill_b, skill_c])

# Retrieve the most relevant skill for a task
results = index.retrieve("handle customer refund request", top_k=3)
for skill, score in results:
    print(f"{skill.name}: {score:.3f}")

# With embedding backend for semantic retrieval
from agent_procedural_memory import EmbeddingBackend

class MyEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        # Wrap any embedding provider here
        ...

index = SkillIndex(embedding_backend=MyEmbedder())
index.add_many(skills)
index.save("skill_index.json")  # Persist for later
```

### Cross-task generalization

```python
from agent_procedural_memory import extract_sub_procedures

# Find reusable sub-procedures shared across different task skills
report = extract_sub_procedures([refund_skill, exchange_skill, escalation_skill])
for sp in report.sub_procedures:
    print(f"{' → '.join(sp.actions)} (found in {sp.support} skills)")
    # Convert to a reusable skill
    fragment_skill = sp.to_skill()
```

### Online learning

```python
from agent_procedural_memory import OnlineLearner

# Start from an existing mined skill
learner = OnlineLearner.from_skill(skill)

# Incrementally refine with new traces as they arrive
for trace in new_traces_stream:
    updated_skill = learner.update(trace)
    # Skill confidence, procedure, and failure modes update incrementally

# Persist and restore learner state
learner.save("learner_state.json")
restored = OnlineLearner.load("learner_state.json")
```

## Run The Demo

```bash
python3 demos/public_traces_demo.py
```

The demo:

1. Loads OpenTelemetry-shaped traces from `data/public_trace_fixtures/support_refund_otel.jsonl`.
2. Generates branch-covering exploratory traces for the same workflow.
3. Mines a procedural skill from both successful and failed traces.
4. Evaluates a baseline agent, a few-shot-success agent, and a skill-memory agent.
5. Writes the generated skill to `artifacts/refund_workflow/SKILL.md`.

If the optional `datasets` package and network are available, the same loader can sample from `Exgentic/agent-llm-traces-v2`, a public OpenTelemetry-shaped trace dataset.

## Code Structure

```
src/agent_procedural_memory/
├── __init__.py          # Public API exports
├── models.py            # Core data structures (Span, AgentTrace, Skill, SkillStep, ...)
├── otel.py              # OpenTelemetry trace normalization and I/O
├── miner.py             # Procedural memory mining (deterministic + LLM-augmented)
├── confidence.py        # Bayesian confidence scoring (Beta-Binomial model)
├── llm.py               # LLM + embedding abstraction layer, prompt templates, retry logic
├── consolidation.py     # Skill merging via Jaccard overlap + LLM judgment
├── retrieval.py         # Embedding-based skill retrieval with Jaccard fallback
├── generalization.py    # Cross-task sub-procedure extraction (n-gram + LLM alignment)
├── online.py            # Online learning — incremental skill refinement from new traces
├── exploration.py       # Workflow graph path enumeration for synthetic traces
├── evaluation.py        # Three-condition evaluation with bootstrap CI + LLM-as-judge
├── toy_agent.py         # Deterministic demo agent and evaluation case builder
└── demo.py              # CLI entry point (apm-demo)

tests/
├── test_library.py      # 18 tests: mining, evaluation, exploration, confidence, consolidation
├── test_retrieval.py    # 14 tests: indexing, retrieval, serialization, similarity helpers
├── test_generalization.py # 10 tests: n-gram extraction, pruning, cross-task filtering
└── test_online.py       # 10 tests: incremental updates, promotion/demotion, serialization

data/
└── public_trace_fixtures/
    └── support_refund_otel.jsonl   # 6 fixture traces (3 success, 3 failure)

artifacts/
└── refund_workflow/
    ├── SKILL.md                    # Generated skill output
    └── exploratory_traces.jsonl    # Generated exploratory traces
```

## Module Details

### `models.py` — Core Data Structures

| Class | Description |
|---|---|
| `Span` | Normalized OpenTelemetry span. Properties: `failed`, `action_name`, `kind`. |
| `AgentTrace` | Ordered spans with workflow metadata. Properties: `succeeded`, `action_sequence`. |
| `SkillStep` | One procedure instruction with `action`, `rationale`, `evidence_count`, `required`, `confidence`, `confidence_interval`, and `depends_on` fields. |
| `Skill` | Portable agent skill with procedure, failure modes, recovery rules, and verification checks. Methods: `required_actions()`, `to_markdown()`. |
| `EvaluationCase` | Test case with expected/distractor actions and task attributes. |
| `RunResult` | Scored execution result per case per condition. |

### `otel.py` — OpenTelemetry Normalization

Handles multiple OTel field naming conventions (camelCase/snake_case), status code normalization, and outcome inference. Functions:

- `normalize_otel_trace(raw_dict) → AgentTrace` — Normalize any OTel-shaped JSON record.
- `load_otel_jsonl(path) → list[AgentTrace]` — Read newline-delimited OTel JSON.
- `save_otel_jsonl(traces, path)` — Persist traces as JSONL.
- `load_public_exgentic(limit, split, fallback_path)` — Load from Hugging Face with local fallback.

### `miner.py` — Procedural Memory Mining

The `ProceduralMemoryMiner` class extracts reusable skills from traces. It operates in two modes:

**Deterministic mode** (no LLM required):

1. Split traces into successes and failures.
2. Count action frequencies across successful traces.
3. Order actions canonically by average position in successful sequences, with alphabetical tiebreaker.
4. Classify each action as required (appears in ≥ `min_success_support` fraction of successes) or optional.
5. Generate template-based rationales from success/failure presence patterns.
6. Mine failure modes: missing steps, error types, retry loops.
7. Generate recovery rules and verification checks.

**LLM-augmented mode** (adds to deterministic):

1. **Semantic action clustering** — Ask LLM to group synonymous action names (e.g., `verify_identity` → `authenticate_customer`). Remap all traces and counts to canonical names.
2. **LLM-generated rationales** — Provide the LLM with frequency evidence, position data, predecessor/successor context, and Bayesian confidence. LLM synthesizes a contextual one-sentence rationale per step.
3. **Causal dependency inference** — Ask LLM to identify which steps causally depend on which (not just correlational ordering). Stored as `metadata["causal_dependencies"]`.
4. **LLM-synthesized descriptions** — Generate natural-language skill descriptions and trigger conditions.

**Configuration:**

| Parameter | Default | Description |
|---|---|---|
| `min_success_support` | `0.5` | Minimum fraction of successes for a step to be "required" |
| `max_steps` | `12` | Maximum procedure length |
| `temporal_decay` | `0.0` | Exponential decay for trace recency (0 = disabled) |
| `llm_config` | `None` | LLM backend configuration (None = deterministic only) |
| `confidence_level` | `0.90` | Credible interval level for Bayesian scoring |

### `confidence.py` — Bayesian Confidence Scoring

Uses the Beta-Binomial conjugate model to estimate step reliability:

- Each action is a Bernoulli trial: appears in successful trace (success) or failed trace (failure).
- Prior: Beta(1, 1) (uninformative / uniform).
- Posterior: Beta(α₀ + s, β₀ + f) where s = successes, f = failures.
- **Posterior mean:** E[θ] = α / (α + β) — point estimate of reliability.
- **Credible interval:** Normal approximation to the Beta posterior (Bernstein-von Mises theorem).
- **Evidence strength:** weak (≤2 observations), moderate (3–10), strong (>10).
- **is_reliable:** True when lower CI bound > 0.5.

The `StepConfidence` dataclass is computed per action by `compute_step_confidence()` and attached to skill metadata.

### `llm.py` — LLM Abstraction Layer

Provides a minimal, provider-agnostic LLM interface:

- **`LLMBackend` Protocol** — Runtime-checkable, single method: `complete(prompt, temperature, max_tokens) → str`. Any provider (Anthropic, OpenAI, local models, test stubs) can implement it.
- **`NullLLMBackend`** — Returns empty strings, triggering deterministic fallback everywhere.
- **`LLMConfig`** — Groups backend, temperature, retry count, and fallback behavior.
- **`llm_complete()`** — Retry-aware completion with structured logging.
- **`llm_complete_json()`** — Parses LLM response as JSON, strips markdown code fences.

**Embedding interface:**

- **`EmbeddingBackend` Protocol** — Runtime-checkable: `embed(texts: list[str]) → list[list[float]]`. Used by `SkillIndex` for semantic retrieval.
- **`NullEmbeddingBackend`** — Returns empty lists, triggering Jaccard fallback in retrieval.

**Prompt templates:**

| Template | Purpose |
|---|---|
| `RATIONALE_PROMPT` | Generate contextual rationale for a procedure step |
| `CAUSAL_PROMPT` | Infer causal dependencies between steps |
| `CLUSTER_PROMPT` | Group synonymous action names |
| `SYNTHESIS_PROMPT` | Generate skill description and triggers |
| `CONSOLIDATION_PROMPT` | Judge whether two skills should merge |
| `JUDGE_PROMPT` | LLM-as-judge evaluation scoring |
| `SUBPROCEDURE_ALIGN_PROMPT` | Judge whether two sub-procedures are semantically equivalent |
| `REFINEMENT_PROMPT` | Re-synthesize rationales/description after online learning updates |

### `consolidation.py` — Skill Consolidation

Merges overlapping skills using agglomerative (bottom-up) clustering:

1. Compute pairwise Jaccard similarity over required action sets.
2. Greedily merge the most similar pair exceeding `jaccard_threshold`.
3. For pairs below the threshold but above 0.1, optionally ask the LLM whether they are semantically equivalent.
4. Merged skills union procedures (keeping higher-evidence variants), failure modes, recovery rules, and verification checks.

**Configuration** via `ConsolidationConfig`:

| Parameter | Default | Description |
|---|---|---|
| `jaccard_threshold` | `0.5` | Minimum Jaccard overlap for automatic merge |
| `use_llm_judgment` | `True` | Enable LLM-based semantic merge for low-overlap pairs |
| `max_merges` | `10` | Safety bound on merge iterations |

### `exploration.py` — Workflow Graph Exploration

The `TraceExplorer` generates synthetic traces that systematically cover all branches of a `WorkflowGraph`:

- **`WorkflowGraph`** — Directed graph with `start`, `terminal`, and `nodes` (each `WorkflowNode` has a name and `next_actions` tuple).
- **Algorithm:** Depth-limited acyclic path enumeration from start to terminal.
- **Output:** `ExplorationReport` with traces, path count, covered actions, and uncovered actions.

### `evaluation.py` — Evaluation Framework

The `EvaluationSuite` compares agent performance across three conditions:

| Condition | What the agent receives |
|---|---|
| `baseline` | No skill, no examples |
| `few_shot_success_traces` | Past successful action sequences |
| `procedural_skill` | Generated Skill object |

**Enhancements over basic comparison:**

- **Bootstrap confidence intervals** — Percentile bootstrap (configurable samples + seed) produces 95% CIs for each condition's average score.
- **LLM-as-judge** — When `llm_judge_config` is provided, results are re-scored by the LLM. Final score blends 60% deterministic + 40% LLM judge.
- **`ConditionMetrics`** — Per-condition dataclass with `avg_score`, `success_rate`, `delta_vs_baseline`, `score_ci`, and `n_cases`.

### `retrieval.py` — Skill Retrieval

The `SkillIndex` stores and retrieves skills by task-context similarity:

**With embeddings** (when `EmbeddingBackend` is provided):
1. On `add(skill)`: concatenate name, description, triggers, and required actions into a text key. Embed via the backend and store the vector.
2. On `retrieve(query, top_k)`: embed the query, compute cosine similarity against all stored embeddings, return top-k.

**Deterministic fallback** (no embedding backend):
1. On `add(skill)`: tokenize the text key into a lowercased word set.
2. On `retrieve(query, top_k)`: tokenize the query, compute Jaccard similarity of token sets, return top-k.

**Key methods:**

| Method | Description |
|---|---|
| `add(skill)` / `add_many(skills)` | Index one or multiple skills |
| `retrieve(query, top_k)` | Return `(Skill, score)` pairs sorted by relevance |
| `remove(skill_name)` | Remove a skill by name |
| `save(path)` / `load(path)` | JSON serialization (embeddings stored as float lists) |
| `size` / `skill_names` | Index introspection properties |

### `generalization.py` — Cross-Task Generalization

Discovers transferable action fragments shared across skills from different tasks.

**Algorithm (contiguous n-gram subsequence mining):**

1. Extract ordered required-action sequences from each skill.
2. Generate all contiguous n-grams of length `min_length..max_length` from each sequence.
3. Count n-grams across skills. Keep those in ≥ `min_support` distinct skills.
4. When `cross_task_only=True` (default), exclude fragments found only in same-task skills.
5. **Maximal pruning:** remove n-grams that are contiguous sub-sequences of a longer frequent n-gram with equal or greater support.
6. **LLM alignment (optional):** present near-miss fragment pairs to the LLM to judge semantic equivalence. Merge confirmed matches.

**Output:** `GeneralizationReport` with `sub_procedures` (each a `SubProcedure` with `actions`, `source_skills`, `support`, `description`). Sub-procedures can be converted to `Skill` objects via `to_skill()`.

| Parameter | Default | Description |
|---|---|---|
| `min_support` | `2` | Minimum number of skills a fragment must appear in |
| `min_length` | `2` | Minimum actions in a fragment |
| `max_length` | `6` | Maximum actions in a fragment |
| `cross_task_only` | `True` | Only count fragments across different task names |

### `online.py` — Online Learning

The `OnlineLearner` incrementally refines an existing `Skill` from new trace evidence without re-mining from scratch.

**Algorithm (per `update(trace)` call):**

1. **Update running counts:** increment success/failure counters for each action in the trace.
2. **Recompute confidence:** rebuild `StepConfidence` (Beta-Binomial posterior) for all actions — O(n) since the conjugate update is a single α or β increment.
3. **Promote new actions:** if a previously unseen action crosses `min_success_support`, add it to the procedure as a required step.
4. **Demote weak actions:** if an existing required step's lower CI bound drops below `demotion_threshold`, mark it optional.
5. **Update failure modes:** if the trace failed, extract new failure patterns (missing steps, error types, retry loops).
6. **LLM refinement (optional):** when confidence deltas exceed 0.1, ask the LLM to re-synthesize rationales and description.
7. **Return new Skill:** construct a new `Skill` with incremented `version`. The learner's internal state is mutable; the skill is immutable.

**Bootstrapping:** `OnlineLearner.from_skill(skill)` extracts initial counts from the skill's metadata, enabling seamless transition from offline mining to online refinement.

**Serialization:** `save(path)` / `load(path)` persist the learner's running statistics (counts, config) as JSON for checkpoint/restore across agent restarts.

| Parameter | Default | Description |
|---|---|---|
| `min_success_support` | `0.5` | Fraction threshold for promoting actions |
| `confidence_level` | `0.90` | Credible interval level |
| `demotion_threshold` | `0.3` | Lower CI bound below which a step is demoted |

### `toy_agent.py` — Demo Agent

`RefundWorkflowAgent` is a deterministic agent demonstrating evaluation mechanics. It produces different action sequences under each condition (baseline/few-shot/skill) and scores them against expected actions with penalties for missing steps, distractor actions, and wrong ordering.

`build_refund_eval_cases()` creates three evaluation cases: standard refund, missing receipt, and manual review.

## Algorithm Design Summary

```
OpenTelemetry Traces (JSONL)
    │ [otel.py: normalize]
    ▼
AgentTrace objects
    │
    ├──► [miner.py + llm.py + confidence.py]  ──  OFFLINE MINING
    │        ├─ Action clustering (LLM, optional)
    │        ├─ Canonical ordering (avg position, statistical)
    │        ├─ Bayesian confidence (Beta-Binomial posterior)
    │        ├─ Rationale generation (LLM with template fallback)
    │        ├─ Causal inference (LLM, optional)
    │        └─► Skill object
    │                │
    │                ├──► [retrieval.py]  ──  SKILL RETRIEVAL
    │                │        ├─ SkillIndex: add / retrieve / remove
    │                │        ├─ Cosine similarity (with EmbeddingBackend)
    │                │        ├─ Jaccard token similarity (deterministic fallback)
    │                │        └─► Ranked (Skill, score) pairs
    │                │
    │                ├──► [generalization.py]  ──  CROSS-TASK TRANSFER
    │                │        ├─ Contiguous n-gram extraction
    │                │        ├─ Maximal subsequence pruning
    │                │        ├─ Cross-task filtering
    │                │        ├─ LLM semantic alignment (optional)
    │                │        └─► SubProcedure fragments → reusable Skills
    │                │
    │                └──► [online.py]  ──  ONLINE LEARNING
    │                         ├─ Incremental Beta-Binomial updates (O(n) per trace)
    │                         ├─ Action promotion / demotion
    │                         ├─ Failure mode accumulation
    │                         ├─ LLM-augmented re-synthesis (optional)
    │                         └─► Refined Skill (version++)
    │
    ├──► [consolidation.py]  ──  SKILL MERGING
    │        ├─ Jaccard overlap detection
    │        ├─ LLM-judged semantic merge (optional)
    │        └─► Consolidated skill set
    │
    ├──► [exploration.py]  ──  TRACE GENERATION
    │        └─► Synthetic traces for uncovered workflow paths
    │
    └──► [evaluation.py]  ──  EVALUATION
             ├─ Three-condition comparison
             ├─ Bootstrap CI (percentile method)
             ├─ LLM-as-judge scoring (optional)
             └─► EvaluationReport with ConditionMetrics
```

**Design principle:** Every LLM/embedding-augmented path has a deterministic fallback. `NullLLMBackend` and `NullEmbeddingBackend` return empty values, and every caller treats empty as "use the deterministic algorithm." The library works with zero external dependencies and no API keys.

## What Counts As OpenTelemetry-Shaped?

The normalizer expects each trace record to contain a `spans` array. Each span may include:

- `name`
- `span_id` / `spanId`
- `parent_span_id` / `parentSpanId`
- `start_time` / `startTimeUnixNano` / `end_time` / `endTimeUnixNano`
- `status` or `status_code` or `otel.status_code`
- `attributes`
- `events`

Common GenAI and workflow attributes such as `gen_ai.operation.name`, `tool.name`, `agent.action`, `task.name`, `task.success`, and `error.type` are recognized, but arbitrary attributes are preserved.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

52 tests across 9 classes in 4 test files:

- **MinerTests** (4) — Skill mining, confidence metadata, deterministic reproducibility, null LLM fallback.
- **EvaluationTests** (3) — Three-condition comparison, bootstrap CIs, report formatting.
- **ExplorationTests** (1) — Path-covering trace generation.
- **ConfidenceTests** (5) — Beta posterior math, evidence strength, z-score lookup.
- **ConsolidationTests** (5) — Jaccard similarity, deterministic merge, consolidation behavior.
- **RetrievalTests** (8) — Indexing, Jaccard retrieval, mock embedding retrieval, save/load roundtrip, removal, embedding failure fallback.
- **SimilarityHelperTests** (6) — Cosine similarity, Jaccard similarity, tokenization.
- **GeneralizationTests** (10) — N-gram extraction, maximal pruning, cross-task filtering, sub-procedure-to-skill conversion.
- **OnlineLearnerTests** (10) — Incremental updates, promotion/demotion, batch consistency, version tracking, serialization roundtrip.
