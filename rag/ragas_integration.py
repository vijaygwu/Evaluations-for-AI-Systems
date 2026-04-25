"""
RAGAS Integration for RAG Evaluation
====================================

Integration with the RAGAS library for comprehensive RAG evaluation.

RAGAS (Retrieval Augmented Generation Assessment) provides standardized
metrics for evaluating RAG pipelines.

Book Reference: Chapter 23 discusses RAG evaluation metrics including
those implemented in RAGAS.

Dependencies:
    pip install ragas langchain langchain-openai
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class RAGASResult:
    """Result from RAGAS evaluation."""
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    answer_correctness: Optional[float] = None
    overall_score: float = 0.0


class RAGASEvaluator:
    """
    Wrapper for RAGAS library evaluation.

    RAGAS Metrics:
    - Faithfulness: How factually consistent is the answer with the context?
    - Answer Relevancy: How relevant is the answer to the question?
    - Context Precision: How relevant are the retrieved contexts?
    - Context Recall: Is all necessary information present in the context?
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        embeddings_model: str = "text-embedding-3-small",
    ):
        """
        Initialize RAGASEvaluator.

        Args:
            model_name: LLM for evaluation
            embeddings_model: Embedding model for semantic comparisons
        """
        self.model_name = model_name
        self.embeddings_model = embeddings_model
        self._ragas_available = None

    def _check_ragas_available(self) -> bool:
        """Check if RAGAS is installed and available."""
        if self._ragas_available is None:
            try:
                import ragas
                self._ragas_available = True
            except ImportError:
                self._ragas_available = False
        return self._ragas_available

    def evaluate(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate RAG outputs using RAGAS metrics.

        Args:
            questions: List of questions
            answers: List of generated answers
            contexts: List of retrieved context lists (one per question)
            ground_truths: Optional list of reference answers

        Returns:
            Dict with RAGAS metrics

        Example:
            >>> evaluator = RAGASEvaluator()
            >>> result = evaluator.evaluate(
            ...     questions=["What is the capital of France?"],
            ...     answers=["Paris is the capital of France."],
            ...     contexts=[["France is a country. Paris is its capital."]],
            ...     ground_truths=["Paris"]
            ... )
        """
        if not self._check_ragas_available():
            return self._fallback_evaluate(questions, answers, contexts, ground_truths)

        try:
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            )
            from datasets import Dataset
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings

            # Prepare data in RAGAS format
            data = {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
            }
            if ground_truths:
                data["ground_truth"] = ground_truths

            dataset = Dataset.from_dict(data)

            # Configure models
            llm = ChatOpenAI(model=self.model_name)
            embeddings = OpenAIEmbeddings(model=self.embeddings_model)

            # Select metrics
            metrics = [
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ]

            # Run evaluation
            result = evaluate(
                dataset,
                metrics=metrics,
                llm=llm,
                embeddings=embeddings,
            )

            return {
                "faithfulness": result["faithfulness"],
                "answer_relevancy": result["answer_relevancy"],
                "context_precision": result["context_precision"],
                "context_recall": result["context_recall"],
                "per_sample": result.to_pandas().to_dict("records"),
            }

        except Exception as e:
            print(f"RAGAS evaluation failed: {e}")
            return self._fallback_evaluate(questions, answers, contexts, ground_truths)

    def _fallback_evaluate(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Fallback evaluation when RAGAS is not available.

        Uses simple heuristics to approximate RAGAS metrics.
        """
        results = {
            "faithfulness": [],
            "answer_relevancy": [],
            "context_precision": [],
            "context_recall": [],
        }

        for i, (question, answer, context_list) in enumerate(zip(questions, answers, contexts)):
            # Combine contexts
            full_context = " ".join(context_list).lower()
            answer_lower = answer.lower()
            question_lower = question.lower()

            # Simple faithfulness: word overlap between answer and context
            answer_words = set(answer_lower.split())
            context_words = set(full_context.split())
            if answer_words:
                faithfulness = len(answer_words & context_words) / len(answer_words)
            else:
                faithfulness = 0.0
            results["faithfulness"].append(faithfulness)

            # Simple answer relevancy: word overlap between answer and question
            question_words = set(question_lower.split()) - {"what", "is", "the", "a", "an", "how", "why", "when", "where"}
            if question_words:
                relevancy = len(answer_words & question_words) / len(question_words)
            else:
                relevancy = 0.5
            results["answer_relevancy"].append(min(1.0, relevancy))

            # Simple context precision: question keywords in context
            if question_words:
                precision = len(context_words & question_words) / len(question_words)
            else:
                precision = 0.5
            results["context_precision"].append(min(1.0, precision))

            # Simple context recall: if ground truth available
            if ground_truths and i < len(ground_truths):
                truth_words = set(ground_truths[i].lower().split())
                if truth_words:
                    recall = len(context_words & truth_words) / len(truth_words)
                else:
                    recall = 0.5
            else:
                recall = 0.5
            results["context_recall"].append(min(1.0, recall))

        # Aggregate
        return {
            "faithfulness": sum(results["faithfulness"]) / len(results["faithfulness"]) if results["faithfulness"] else 0.0,
            "answer_relevancy": sum(results["answer_relevancy"]) / len(results["answer_relevancy"]) if results["answer_relevancy"] else 0.0,
            "context_precision": sum(results["context_precision"]) / len(results["context_precision"]) if results["context_precision"] else 0.0,
            "context_recall": sum(results["context_recall"]) / len(results["context_recall"]) if results["context_recall"] else 0.0,
            "note": "Using fallback heuristic evaluation (RAGAS not available)",
            "per_sample": [
                {
                    "question": questions[i],
                    "faithfulness": results["faithfulness"][i],
                    "answer_relevancy": results["answer_relevancy"][i],
                    "context_precision": results["context_precision"][i],
                    "context_recall": results["context_recall"][i],
                }
                for i in range(len(questions))
            ],
        }

    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> RAGASResult:
        """
        Evaluate a single RAG response.

        Convenience method for evaluating one question at a time.

        Args:
            question: The question asked
            answer: The generated answer
            contexts: Retrieved context documents
            ground_truth: Optional reference answer

        Returns:
            RAGASResult with all metrics
        """
        ground_truths = [ground_truth] if ground_truth else None

        result = self.evaluate(
            questions=[question],
            answers=[answer],
            contexts=[contexts],
            ground_truths=ground_truths,
        )

        return RAGASResult(
            faithfulness=result.get("faithfulness", 0.0),
            answer_relevancy=result.get("answer_relevancy", 0.0),
            context_precision=result.get("context_precision", 0.0),
            context_recall=result.get("context_recall", 0.0),
            overall_score=(
                result.get("faithfulness", 0.0) +
                result.get("answer_relevancy", 0.0) +
                result.get("context_precision", 0.0) +
                result.get("context_recall", 0.0)
            ) / 4,
        )


def create_rag_evaluation_dataset(
    questions: List[str],
    contexts: List[List[str]],
    answers: List[str],
    ground_truths: Optional[List[str]] = None,
) -> Dict[str, List]:
    """
    Create a dataset suitable for RAG evaluation.

    Args:
        questions: List of questions
        contexts: List of context lists
        answers: List of generated answers
        ground_truths: Optional list of reference answers

    Returns:
        Dict formatted for RAGAS evaluation
    """
    dataset = {
        "question": questions,
        "contexts": contexts,
        "answer": answers,
    }

    if ground_truths:
        dataset["ground_truth"] = ground_truths

    return dataset


# Demo usage
if __name__ == "__main__":
    print("=" * 60)
    print("Chapter 23: RAGAS Integration Demo")
    print("=" * 60)

    # Example dataset
    questions = [
        "What is the capital of France?",
        "When was the Eiffel Tower built?",
        "What is the population of France?",
    ]

    contexts = [
        ["France is a country in Western Europe. Paris is the capital city of France."],
        ["The Eiffel Tower is a wrought-iron lattice tower in Paris. It was constructed from 1887 to 1889."],
        ["France has a population of approximately 67 million people as of 2023."],
    ]

    answers = [
        "Paris is the capital of France.",
        "The Eiffel Tower was built between 1887 and 1889.",
        "France has about 67 million people.",
    ]

    ground_truths = [
        "Paris",
        "1889",
        "67 million",
    ]

    print("\n1. Sample RAG Evaluation Dataset")
    print("-" * 40)
    for i, (q, a, c) in enumerate(zip(questions, answers, contexts)):
        print(f"\nQ{i+1}: {q}")
        print(f"A{i+1}: {a}")
        print(f"Context: {c[0][:50]}...")

    print("\n2. RAGAS Evaluation")
    print("-" * 40)

    evaluator = RAGASEvaluator()
    results = evaluator.evaluate(
        questions=questions,
        answers=answers,
        contexts=contexts,
        ground_truths=ground_truths,
    )

    print("\nAggregate Metrics:")
    print(f"  Faithfulness: {results['faithfulness']:.3f}")
    print(f"  Answer Relevancy: {results['answer_relevancy']:.3f}")
    print(f"  Context Precision: {results['context_precision']:.3f}")
    print(f"  Context Recall: {results['context_recall']:.3f}")

    if "note" in results:
        print(f"\n  Note: {results['note']}")

    print("\n3. Single Response Evaluation")
    print("-" * 40)

    single_result = evaluator.evaluate_single(
        question="What is the capital of France?",
        answer="Paris is the beautiful capital of France, known as the City of Light.",
        contexts=["France is a country in Europe. Paris is its capital city."],
        ground_truth="Paris",
    )

    print(f"Question: What is the capital of France?")
    print(f"Answer: Paris is the beautiful capital of France...")
    print(f"\nMetrics:")
    print(f"  Faithfulness: {single_result.faithfulness:.3f}")
    print(f"  Answer Relevancy: {single_result.answer_relevancy:.3f}")
    print(f"  Context Precision: {single_result.context_precision:.3f}")
    print(f"  Context Recall: {single_result.context_recall:.3f}")
    print(f"  Overall Score: {single_result.overall_score:.3f}")

    print("\n4. Interpreting Results (Target Scores from Chapter 23)")
    print("-" * 40)
    print("Metric              Score   Target   Status")
    print("-" * 40)

    targets = [
        ("Faithfulness", results['faithfulness'], 0.9),
        ("Answer Relevancy", results['answer_relevancy'], 0.9),
        ("Context Precision", results['context_precision'], 0.8),
        ("Context Recall", results['context_recall'], 0.9),
    ]

    for name, score, target in targets:
        status = "PASS" if score >= target else "FAIL"
        print(f"{name:<20} {score:.3f}   {target:.1f}     {status}")

    print("\n" + "=" * 60)
    print("Key Takeaway: RAGAS provides standardized metrics for RAG:")
    print("- Faithfulness: Is the answer grounded in context?")
    print("- Answer Relevancy: Does the answer address the question?")
    print("- Context Precision: Is the retrieved context relevant?")
    print("- Context Recall: Is all needed information retrieved?")
    print("=" * 60)
