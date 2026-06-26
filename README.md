# Beyond Vibes — Companion Code

[![CI](https://github.com/vijaygwu/Evaluations-for-AI-Systems/actions/workflows/ci.yml/badge.svg)](https://github.com/vijaygwu/Evaluations-for-AI-Systems/actions/workflows/ci.yml)

Working, tested Python implementations of the evaluation techniques in
**_Beyond Vibes: Evaluating Generative and Agentic AI So You Can Ship Systems
You Trust_** (Book 6 of *The AI Engineer's Library*, by Dr. Vijay Raghavan).

The book shows **illustrative excerpts**; this repository is the **runnable,
tested counterpart**. Where a chapter sketches an idea, the corresponding
module here is the full, working version with error handling and tests.

---

## Repository structure

```
.
├── ch03_graders/                # Ch 3 — Grader taxonomy
│   ├── exact_match.py           #   String / regex / JSON match graders
│   ├── semantic_similarity.py   #   Embedding-based similarity grader
│   ├── llm_as_judge.py          #   LLM-as-judge: LLMJudge, RubricGrader, SimpleQAGrader
│   ├── code_execution.py        #   Execution-based grading (pass@k, HumanEval)
│   └── advanced_graders.py      #   Multi-aspect, confidence-weighted, contrastive, structured
├── ch04_statistics/             # Ch 4 — Statistical rigor
│   ├── sample_size.py           #   Sample-size & power calculations
│   ├── confidence_intervals.py  #   CI computation
│   └── significance_tests.py    #   compare_proportions, paired t, McNemar, permutation
├── ch08_agents/                 # Ch 8 — Agent evaluation
│   ├── trajectory_scorer.py     #   Multi-step trajectory scoring
│   └── tool_use_eval.py         #   Tool-call verification
├── ch13_safety/                 # Ch 13 — Red-teaming & safety
│   ├── red_team_patterns.py     #   Attack-pattern detection, model robustness
│   ├── prompt_injection.py      #   Injection-defense testing
│   └── robustness.py            #   Obfuscation attacks + ASR / robustness / FPR metrics
├── ch23_rag/                    # Ch 23 — RAG evaluation
│   ├── retrieval_metrics.py     #   Precision@k, Recall@k, MRR, NDCG, MAP
│   ├── faithfulness.py          #   FaithfulnessScorer, GroundednessChecker, CitationChecker
│   ├── groundedness.py          #   evaluate_groundedness, detect_rag_hallucinations
│   └── ragas_integration.py     #   RAGAS library wrapper (optional dependency)
├── utils/
│   └── llm_clients.py           #   Provider-agnostic call_llm() wrapper
├── tests/
│   └── test_execution.py        #   Mocked end-to-end execution tests (no API keys)
├── requirements.txt
├── setup.py
└── .github/workflows/ci.yml     #   Continuous integration
```

> The code lives at the repository root. It is also vendored into the book's
> manuscript repo under `book6-latex/companion-code/`; the two are kept in sync.

---

## Installation

**Full install** (everything, for live API usage):

```bash
pip install -r requirements.txt
```

**Minimal install** (just to run the offline test suite / deterministic
graders — heavy libs are lazy-loaded, so you don't need them for this):

```bash
pip install numpy scipy
```

`python_requires >= 3.9`; CI tests on 3.10 / 3.11 / 3.12.

### What runs offline vs. what needs API keys

| Works with no keys / minimal deps | Needs an API key or an optional library |
|---|---|
| Exact / regex / JSON match graders | `LLMJudge`, `RubricGrader`, `SimpleQAGrader` (LLM call) |
| Retrieval metrics (P@k, R@k, MRR, NDCG, MAP) | `FaithfulnessScorer`, `GroundednessChecker`, `CitationChecker` (LLM call) |
| Sample size, CIs, significance tests | `SemanticSimilarityGrader` (needs `sentence-transformers`) |
| Multi-aspect / confidence-weighted / contrastive graders | `RAGASEvaluator` (needs `ragas`) |
| ch13 obfuscation + robustness metrics | — |
| `evaluate_groundedness` (lexical) | `detect_rag_hallucinations` (needs an LLM judge callable) |

For the LLM-backed graders, set your keys:

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

---

## Quick start

### Graders (Chapter 3)

```python
from ch03_graders.exact_match import ExactMatchGrader
from ch03_graders.llm_as_judge import LLMJudge

# Exact match returns a GradeResult (.score, .passed, .explanation)
result = ExactMatchGrader(normalize=True).grade("POSITIVE", "positive")
print(result.score, result.passed)   # 1.0 True

# LLM-as-judge for subjective quality (needs an API key)
judge = LLMJudge(rubric="helpfulness")
verdict = judge.grade(
    question="How do I reset my password?",
    response="Click 'Forgot Password' on the login page...",
)
print(verdict.grade, verdict.score)
```

### Statistical rigor (Chapter 4)

```python
from ch04_statistics.sample_size import calculate_sample_size
from ch04_statistics.significance_tests import compare_proportions

n = calculate_sample_size(baseline=0.80, mde=0.05, power=0.80)

result = compare_proportions(
    successes_a=160, total_a=200,   # 80%
    successes_b=176, total_b=200,   # 88%
)
print(result.p_value)
```

### RAG evaluation (Chapter 23)

```python
from ch23_rag.retrieval_metrics import precision_at_k, recall_at_k, mean_reciprocal_rank
from ch23_rag.groundedness import evaluate_groundedness

retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
relevant = {"doc1", "doc3", "doc7"}
print(precision_at_k(retrieved, relevant, k=5))   # 0.4
print(mean_reciprocal_rank(retrieved, relevant))  # 1.0

# Lexical groundedness — runs with no API key
g = evaluate_groundedness(
    "Paris is the capital of France.",
    sources=["France's capital city is Paris."],
)
print(g["groundedness_score"])
```

### Agent trajectories (Chapter 8)

```python
from ch08_agents.trajectory_scorer import TrajectoryScorer

scorer = TrajectoryScorer()
result = scorer.score(
    goal="Get current weather in New York",
    trajectory=[
        {"action": "search", "params": {"query": "weather NYC"}, "result": "72F sunny"},
        {"action": "respond", "content": "The weather in NYC is 72F and sunny."},
    ],
)
```

---

## Testing

The suite is layered so that **everything except live API calls runs offline**:

| Layer | What it checks | How to run |
|---|---|---|
| **Syntax** | every module compiles (`py_compile`) | `python -m py_compile $(find . -name '*.py')` |
| **Import smoke** | every module imports without the heavy libs | (see CI, or `python -c "import ch03_graders, ..."`) |
| **Execution (mocked)** | the LLM- / embedder-backed functions run **end to end** with no keys | `python tests/test_execution.py` |

### Mocked execution tests (`tests/test_execution.py`)

API calls can't run in CI (keys, cost, nondeterminism), so the tests **inject
fakes at the seams** and drive each function through its *real* success path:

- a **fake `call_llm`** returning a canned `LLMResponse` → exercises
  `RubricGrader`, `SimpleQAGrader`, `GroundednessChecker.check`,
  `FaithfulnessScorer.score`;
- a **fake embedder** (`.encode`) → exercises `SemanticSimilarityGrader`
  without `sentence-transformers`;
- a **fake `model.generate`** → exercises the ch13 robustness metrics
  (`attack_success_rate`, `robustness_score`, `false_positive_rate`);
- the **RAGAS** path is skipped cleanly when `ragas` is not installed.

```bash
python tests/test_execution.py
# -> PASS: all LLM-/embedder-path functions execute end to end with mocked clients.
```

### Continuous integration

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on every push and
pull request to `main`, across Python 3.10 / 3.11 / 3.12:

1. `py_compile` every module (syntax),
2. import every module (no heavy deps),
3. run `tests/test_execution.py` (mocked end-to-end execution).

The deterministic graders (exact match, retrieval metrics, sample sizing, ch13
obfuscations) are additionally covered by the book's evaluation gauntlet
(`grader_canaries`) in the manuscript repo.

---

## Chapter coverage

| Chapter | Topic | Modules |
|---|---|---|
| 3 | Grader taxonomy | `exact_match`, `semantic_similarity`, `llm_as_judge`, `code_execution`, `advanced_graders` |
| 4 | Statistical rigor | `sample_size`, `confidence_intervals`, `significance_tests` |
| 8 | Agent evaluation | `trajectory_scorer`, `tool_use_eval` |
| 13 | Red-teaming & safety | `red_team_patterns`, `prompt_injection`, `robustness` |
| 23 | RAG evaluation | `retrieval_metrics`, `faithfulness`, `groundedness`, `ragas_integration` |

The other chapters of the book are conceptual or use techniques covered by the
modules above; this package targets the chapters with substantial, reusable code.

---

## Contributing

This is companion code for the book. Issues and PRs that fix bugs or improve
the examples are welcome; please keep changes consistent with the book's
listings. See the main book repository for broader guidelines.

## License

MIT License — see the `LICENSE` file for details.
