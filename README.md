# Beyond Vibes - Companion Code

This repository contains working Python code examples for "Beyond Vibes" (Book 6 in The AI Engineer's Library series).

## Repository Structure

```
companion-code/
├── ch03_graders/            # Grader taxonomy
│   ├── exact_match.py       # String matching graders
│   ├── semantic_similarity.py # Embedding-based graders
│   ├── llm_as_judge.py      # Model-based grading
│   └── code_execution.py    # Execution-based grading
├── ch04_statistics/         # Statistical rigor
│   ├── sample_size.py       # Sample size calculations
│   ├── confidence_intervals.py # CI computation
│   └── significance_tests.py # Hypothesis testing
├── ch08_agents/             # Agent evaluation
│   ├── trajectory_scorer.py # Multi-step scoring
│   └── tool_use_eval.py     # Tool call verification
├── ch13_safety/             # Safety evaluation
│   ├── red_team_patterns.py # Attack pattern detection
│   └── prompt_injection.py  # Injection defense testing
├── ch23_rag/                # RAG evaluation
│   ├── retrieval_metrics.py # Precision, recall, MRR
│   ├── faithfulness.py      # Answer faithfulness scoring
│   └── ragas_integration.py # RAGAS library usage
└── utils/                   # Shared utilities
    ├── __init__.py
    └── llm_clients.py       # API client wrappers
```

## Installation

```bash
pip install -r requirements.txt
```

## Required Dependencies

```
openai>=1.0.0
anthropic>=0.18.0
sentence-transformers>=2.2.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.2.0
ragas>=0.1.0
langchain>=0.1.0
pandas>=2.0.0
```

## Environment Variables

Set up your API keys:

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

## Quick Start

### 1. Basic Graders (Chapter 3)

```python
from ch03_graders.exact_match import ExactMatchGrader
from ch03_graders.semantic_similarity import SemanticSimilarityGrader
from ch03_graders.llm_as_judge import LLMJudge

# Exact match for classification tasks
grader = ExactMatchGrader(normalize=True)
score = grader.grade("POSITIVE", "positive")  # 1.0

# Semantic similarity for paraphrase detection
grader = SemanticSimilarityGrader(threshold=0.8)
score = grader.grade(
    "The cat sat on the mat",
    "A feline rested on the rug"
)

# LLM-as-judge for subjective quality
judge = LLMJudge(rubric="helpfulness")
result = judge.grade(
    question="How do I reset my password?",
    response="Click 'Forgot Password' on the login page..."
)
```

### 2. Statistical Analysis (Chapter 4)

```python
from ch04_statistics.sample_size import calculate_sample_size
from ch04_statistics.significance_tests import compare_proportions

# How many samples to detect a 5% improvement?
n = calculate_sample_size(baseline=0.80, mde=0.05, power=0.80)

# Is the improvement statistically significant?
result = compare_proportions(
    successes_a=160, total_a=200,  # 80% accuracy
    successes_b=176, total_b=200   # 88% accuracy
)
print(f"p-value: {result.p_value:.4f}")
```

### 3. RAG Evaluation (Chapter 23)

```python
from ch23_rag.retrieval_metrics import precision_at_k, recall_at_k, mrr
from ch23_rag.faithfulness import FaithfulnessScorer

# Retrieval quality
retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
relevant = {"doc1", "doc3", "doc7"}
print(f"Precision@5: {precision_at_k(retrieved, relevant, k=5)}")
print(f"Recall@5: {recall_at_k(retrieved, relevant, k=5)}")

# Answer faithfulness
scorer = FaithfulnessScorer()
result = scorer.score(
    question="What is the capital of France?",
    answer="Paris is the capital of France.",
    context="France is a country in Europe. Paris is its capital city."
)
```

### 4. Agent Trajectory Evaluation (Chapter 8)

```python
from ch08_agents.trajectory_scorer import TrajectoryScorer

scorer = TrajectoryScorer()
trajectory = [
    {"action": "search", "params": {"query": "weather NYC"}, "result": "72F sunny"},
    {"action": "respond", "content": "The weather in NYC is 72F and sunny."}
]
result = scorer.score(
    goal="Get current weather in New York",
    trajectory=trajectory
)
```

## Running Examples

Each module can be run standalone:

```bash
# Run grader examples
python -m ch03_graders.exact_match
python -m ch03_graders.llm_as_judge

# Run statistical examples
python -m ch04_statistics.sample_size

# Run RAG evaluation examples
python -m ch23_rag.retrieval_metrics
```

## Chapter Coverage

| Chapter | Topic | Examples |
|---------|-------|----------|
| 3 | Grader Taxonomy | Exact, semantic, LLM-as-judge |
| 4 | Statistical Rigor | Sample size, CIs, hypothesis tests |
| 8 | Agent Evaluation | Trajectory scoring, tool use |
| 13 | Red-Teaming | Attack patterns, injection tests |
| 23 | RAG Evaluation | Retrieval metrics, faithfulness |

## Contributing

See the main book repository for contribution guidelines.

## License

MIT License - See LICENSE file for details.
