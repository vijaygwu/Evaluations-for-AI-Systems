# Evaluations for AI Systems

**Companion code repository for Book 6 of The AI Engineer's Library**

This repository contains runnable code examples, grader implementations, statistical utilities, and eval templates from the book "Evaluations for AI Systems."

## Installation

```bash
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

## Repository Structure

```
.
├── graders/                    # Chapter 3: Grader implementations
│   ├── code_based/            # Exact match, fuzzy match, regex, JSON
│   ├── model_based/           # LLM-as-judge, rubric graders
│   └── hybrid/                # Cascade graders
│
├── statistics/                 # Chapter 4: Statistical utilities
│   ├── sample_size.py         # NIST formula implementation
│   ├── confidence_intervals.py
│   └── clustering.py
│
├── datasets/                   # Chapter 5: Dataset utilities
│   ├── stratified_sampler.py
│   └── contamination_check.py
│
├── flywheel/                   # Chapter 6: Eval flywheel
│   ├── runner.py
│   └── transcript_analyzer.py
│
├── agents/                     # Chapters 8-9: Agent evaluation
│   ├── trajectory_scorer.py
│   └── tool_use_evaluator.py
│
├── safety/                     # Chapters 13-15: Safety evaluation
│   ├── red_team_harness.py
│   └── sabotage_detector.py
│
├── templates/                  # Appendix C: Eval templates
│   ├── code_grader_template.yaml
│   ├── llm_judge_prompt.txt
│   └── agent_trajectory_eval.yaml
│
├── examples/                   # Worked examples from chapters
│   └── ...
│
└── tests/                      # Test suite
    └── ...
```

## Quick Start

### Code-Based Graders (Chapter 3)

```python
from graders.code_based import ExactMatchGrader, FuzzyMatchGrader

# Exact match
grader = ExactMatchGrader()
result = grader.grade("42", references=["42", "forty-two"])
print(result)  # {'pass': True, 'score': 1.0, 'matched_reference': '42'}

# Fuzzy match
grader = FuzzyMatchGrader()
result = grader.grade("The answer is 42", references=["42"])
print(result)  # {'pass': True, 'score': 1.0, ...}
```

### Sample Size Calculator (Chapter 4)

```python
from statistics.sample_size import nist_sample_size, margin_of_error_sample_size

# How many samples to detect 10% -> 20% shift?
n = nist_sample_size(p0=0.10, p1=0.20)
print(f"Required N = {n}")  # ~102

# How many samples for ±5% margin of error?
n = margin_of_error_sample_size(margin=0.05)
print(f"Required N = {n}")  # ~385
```

### LLM Judge (Chapter 3)

```python
from graders.model_based import LLMJudge

judge = LLMJudge(model="claude-3-sonnet-20240229")
result = judge.grade(
    task="Answer the user's question helpfully",
    user_query="What is the capital of France?",
    response="The capital of France is Paris.",
    rubric=["accuracy", "completeness", "clarity"]
)
print(result)  # {'score': 5, 'reasoning': '...'}
```

## Book Reference

Each module corresponds to chapters in the book:

| Module | Chapter |
|--------|---------|
| `graders/` | Ch 3: Grader Taxonomy |
| `statistics/` | Ch 4: Statistical Rigor |
| `datasets/` | Ch 5: Eval Dataset Design |
| `flywheel/` | Ch 6: The Eval Flywheel |
| `agents/` | Ch 8-9: Agent Evaluation |
| `safety/` | Ch 13-15: Safety & Red-Teaming |
| `templates/` | Appendix C: Eval Templates |

## License

MIT License - see LICENSE file.

## Author

Vijay Raghavan

Part of **The AI Engineer's Library** series.
