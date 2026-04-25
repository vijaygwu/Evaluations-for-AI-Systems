"""
RAG Faithfulness and Groundedness Evaluation
============================================

Evaluate whether RAG responses are faithful to retrieved context.

Book Reference: Chapter 23, Section "The Faithfulness-Relevance Tradeoff"

Key Concepts:
- Faithfulness: Does the response accurately reflect retrieved documents?
- Groundedness: Is the output derived from retrieved docs (not hallucinated)?
- Citation Accuracy: Are sources correctly attributed?

"High faithfulness means the system only states what the documents say,
though this risks missing relevant information the model possesses."
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

# Import from our utils module
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.llm_clients import call_llm


@dataclass
class FaithfulnessResult:
    """Result of faithfulness evaluation."""
    score: float  # 0.0 to 1.0
    claims_total: int
    claims_supported: int
    claims_contradicted: int
    claims_unverifiable: int
    claim_details: List[Dict[str, Any]]
    explanation: str


@dataclass
class GroundednessResult:
    """Result of groundedness check."""
    score: float  # 0.0 to 1.0
    grounded_sentences: int
    total_sentences: int
    hallucinated_content: List[str]
    explanation: str


class FaithfulnessScorer:
    """
    Evaluate faithfulness of RAG responses to source documents.

    Book Reference: Chapter 23 - "Faithfulness measures whether the response
    accurately reflects retrieved documents."

    Method:
    1. Extract claims from the response
    2. Check each claim against the context
    3. Categorize as supported, contradicted, or unverifiable
    """

    CLAIM_EXTRACTION_PROMPT = """
Extract all factual claims from the following response.
A claim is a statement that can be verified as true or false.

Response: {response}

List each claim on a separate line, prefixed with "CLAIM: "
Only include factual statements, not opinions or hedged statements.
"""

    CLAIM_VERIFICATION_PROMPT = """
Determine if the following claim is supported by the given context.

Context:
{context}

Claim: {claim}

Respond with one of:
- SUPPORTED: The claim is directly stated or clearly implied by the context
- CONTRADICTED: The claim conflicts with information in the context
- UNVERIFIABLE: The context does not contain enough information to verify the claim

VERDICT: [SUPPORTED/CONTRADICTED/UNVERIFIABLE]
REASONING: [Brief explanation]
"""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        provider: str = "openai",
    ):
        """
        Initialize FaithfulnessScorer.

        Args:
            model: LLM model for claim verification
            provider: API provider
        """
        self.model = model
        self.provider = provider

    def extract_claims(self, response: str) -> List[str]:
        """
        Extract factual claims from a response.

        Args:
            response: The RAG response to analyze

        Returns:
            List of extracted claims
        """
        prompt = self.CLAIM_EXTRACTION_PROMPT.format(response=response)

        try:
            result = call_llm(
                prompt=prompt,
                model=self.model,
                provider=self.provider,
                temperature=0.0,
            )

            # Parse claims from response
            claims = []
            for line in result.content.split("\n"):
                if line.strip().startswith("CLAIM:"):
                    claim = line.replace("CLAIM:", "").strip()
                    if claim:
                        claims.append(claim)

            return claims

        except Exception:
            # Fallback: split response into sentences as claims
            sentences = re.split(r'[.!?]+', response)
            return [s.strip() for s in sentences if len(s.strip()) > 10]

    def verify_claim(
        self,
        claim: str,
        context: str,
    ) -> Tuple[str, str]:
        """
        Verify a single claim against context.

        Args:
            claim: The claim to verify
            context: The source context

        Returns:
            Tuple of (verdict, reasoning)
        """
        prompt = self.CLAIM_VERIFICATION_PROMPT.format(
            context=context,
            claim=claim
        )

        try:
            result = call_llm(
                prompt=prompt,
                model=self.model,
                provider=self.provider,
                temperature=0.0,
            )

            # Parse verdict
            content = result.content.upper()
            if "SUPPORTED" in content:
                verdict = "SUPPORTED"
            elif "CONTRADICTED" in content:
                verdict = "CONTRADICTED"
            else:
                verdict = "UNVERIFIABLE"

            # Extract reasoning
            reasoning_match = re.search(
                r"REASONING:\s*(.+)",
                result.content,
                re.IGNORECASE | re.DOTALL
            )
            reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

            return verdict, reasoning

        except Exception as e:
            return "UNVERIFIABLE", f"Error during verification: {str(e)}"

    def score(
        self,
        question: str,
        answer: str,
        context: str,
        use_llm: bool = True,
    ) -> FaithfulnessResult:
        """
        Score the faithfulness of an answer to its context.

        Args:
            question: The original question
            answer: The RAG system's answer
            context: The retrieved context used to generate the answer
            use_llm: Whether to use LLM for claim extraction/verification

        Returns:
            FaithfulnessResult with detailed scoring

        Example:
            >>> scorer = FaithfulnessScorer()
            >>> result = scorer.score(
            ...     question="What is the capital of France?",
            ...     answer="Paris is the capital of France.",
            ...     context="France is a country in Europe. Paris is its capital."
            ... )
            >>> print(f"Faithfulness: {result.score:.2f}")
        """
        if use_llm:
            # Extract claims using LLM
            claims = self.extract_claims(answer)
        else:
            # Simple sentence splitting
            claims = [s.strip() for s in re.split(r'[.!?]+', answer) if len(s.strip()) > 10]

        if not claims:
            return FaithfulnessResult(
                score=1.0,
                claims_total=0,
                claims_supported=0,
                claims_contradicted=0,
                claims_unverifiable=0,
                claim_details=[],
                explanation="No verifiable claims found in response"
            )

        # Verify each claim
        claim_details = []
        supported = 0
        contradicted = 0
        unverifiable = 0

        for claim in claims:
            if use_llm:
                verdict, reasoning = self.verify_claim(claim, context)
            else:
                # Simple heuristic: check if key words appear in context
                claim_words = set(claim.lower().split())
                context_words = set(context.lower().split())
                overlap = len(claim_words & context_words) / len(claim_words) if claim_words else 0

                if overlap > 0.5:
                    verdict = "SUPPORTED"
                    reasoning = f"Word overlap: {overlap:.0%}"
                else:
                    verdict = "UNVERIFIABLE"
                    reasoning = f"Low word overlap: {overlap:.0%}"

            claim_details.append({
                "claim": claim,
                "verdict": verdict,
                "reasoning": reasoning,
            })

            if verdict == "SUPPORTED":
                supported += 1
            elif verdict == "CONTRADICTED":
                contradicted += 1
            else:
                unverifiable += 1

        # Calculate score
        # Supported claims are good, contradicted are bad, unverifiable are neutral
        total = len(claims)
        if total == 0:
            score = 1.0
        else:
            # Score: (supported - contradicted) / total, normalized to [0, 1]
            raw_score = (supported - contradicted) / total
            score = (raw_score + 1) / 2  # Map [-1, 1] to [0, 1]

        explanation_parts = [
            f"{supported}/{total} claims supported",
            f"{contradicted}/{total} contradicted",
            f"{unverifiable}/{total} unverifiable",
        ]

        return FaithfulnessResult(
            score=score,
            claims_total=total,
            claims_supported=supported,
            claims_contradicted=contradicted,
            claims_unverifiable=unverifiable,
            claim_details=claim_details,
            explanation="; ".join(explanation_parts)
        )


class GroundednessChecker:
    """
    Check if response is grounded in context (not hallucinated).

    Book Reference: Chapter 23 - "Groundedness measures whether output
    is derived from retrieved docs."
    """

    GROUNDEDNESS_PROMPT = """
Analyze the following response and determine which parts are grounded in the provided context vs. potentially hallucinated.

Context:
{context}

Response:
{response}

For each sentence in the response, determine if it is:
- GROUNDED: Information can be traced to the context
- HALLUCINATED: Information appears to be made up or from external knowledge

Format your response as:
SENTENCE: [sentence]
STATUS: [GROUNDED/HALLUCINATED]
EVIDENCE: [quote from context if grounded, or "none" if hallucinated]

Finally, provide:
OVERALL_SCORE: [0.0-1.0 representing fraction of grounded content]
"""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        provider: str = "openai",
    ):
        self.model = model
        self.provider = provider

    def check(
        self,
        response: str,
        context: str,
    ) -> GroundednessResult:
        """
        Check groundedness of response in context.

        Args:
            response: The RAG response
            context: The retrieved context

        Returns:
            GroundednessResult with grounded/hallucinated breakdown
        """
        prompt = self.GROUNDEDNESS_PROMPT.format(
            context=context,
            response=response
        )

        try:
            result = call_llm(
                prompt=prompt,
                model=self.model,
                provider=self.provider,
                temperature=0.0,
            )

            # Parse response
            content = result.content

            # Count grounded vs hallucinated
            grounded_count = content.upper().count("STATUS: GROUNDED")
            hallucinated_count = content.upper().count("STATUS: HALLUCINATED")
            total = grounded_count + hallucinated_count

            # Extract overall score
            score_match = re.search(r"OVERALL_SCORE:\s*([\d.]+)", content)
            if score_match:
                score = float(score_match.group(1))
            elif total > 0:
                score = grounded_count / total
            else:
                score = 1.0

            # Extract hallucinated content
            hallucinated = []
            pattern = r"SENTENCE:\s*(.+?)\nSTATUS:\s*HALLUCINATED"
            for match in re.finditer(pattern, content, re.IGNORECASE):
                hallucinated.append(match.group(1).strip())

            return GroundednessResult(
                score=score,
                grounded_sentences=grounded_count,
                total_sentences=total,
                hallucinated_content=hallucinated,
                explanation=f"{grounded_count}/{total} sentences grounded"
            )

        except Exception as e:
            return GroundednessResult(
                score=0.5,
                grounded_sentences=0,
                total_sentences=0,
                hallucinated_content=[],
                explanation=f"Error during check: {str(e)}"
            )


class CitationChecker:
    """
    Check if citations in a response are accurate.

    Book Reference: Chapter 23 - "Citation Accuracy: Sources correctly attributed
    (target >0.95)"
    """

    def __init__(self):
        pass

    def check_citations(
        self,
        response: str,
        documents: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Verify citations in a response against source documents.

        Args:
            response: Response with citations like [1], [doc1], etc.
            documents: Dict mapping citation keys to document content

        Returns:
            Dict with citation accuracy metrics

        Example:
            >>> checker = CitationChecker()
            >>> response = "Paris is the capital [1]. It has the Eiffel Tower [2]."
            >>> docs = {
            ...     "1": "Paris is the capital city of France.",
            ...     "2": "The Eiffel Tower is in Paris."
            ... }
            >>> result = checker.check_citations(response, docs)
        """
        # Extract citations from response
        citation_pattern = r'\[(\w+)\]'
        citations_used = set(re.findall(citation_pattern, response))

        results = {
            "total_citations": len(citations_used),
            "valid_citations": 0,
            "invalid_citations": [],
            "citation_details": [],
        }

        for citation in citations_used:
            if citation in documents:
                results["valid_citations"] += 1
                results["citation_details"].append({
                    "citation": citation,
                    "valid": True,
                    "document_found": True,
                })
            else:
                results["invalid_citations"].append(citation)
                results["citation_details"].append({
                    "citation": citation,
                    "valid": False,
                    "document_found": False,
                })

        # Calculate accuracy
        if results["total_citations"] > 0:
            results["accuracy"] = results["valid_citations"] / results["total_citations"]
        else:
            results["accuracy"] = 1.0  # No citations = no errors

        return results


# Demo usage
if __name__ == "__main__":
    print("=" * 60)
    print("Chapter 23: Faithfulness Evaluation Demo")
    print("=" * 60)

    # Example 1: Basic faithfulness (without LLM)
    print("\n1. Basic Faithfulness Check (Heuristic)")
    print("-" * 40)

    context = """
    France is a country in Western Europe. Paris is the capital city of France.
    The Eiffel Tower, built in 1889, is located in Paris. France has a population
    of approximately 67 million people. The official language is French.
    """

    # Faithful response
    faithful_answer = """
    Paris is the capital of France. The Eiffel Tower is located in Paris
    and was built in 1889. France has about 67 million people.
    """

    # Unfaithful response (contains hallucination)
    unfaithful_answer = """
    Paris is the capital of France. The Eiffel Tower was designed by
    Gustave Eiffel and is 324 meters tall. France won the World Cup in 2018.
    """

    scorer = FaithfulnessScorer()

    print("Context:")
    print(f"  {context[:100]}...")

    print("\nFaithful response:")
    print(f"  {faithful_answer[:80]}...")
    result_faithful = scorer.score(
        question="Tell me about France",
        answer=faithful_answer,
        context=context,
        use_llm=False  # Use heuristic for demo
    )
    print(f"  Faithfulness score: {result_faithful.score:.2f}")
    print(f"  {result_faithful.explanation}")

    print("\nUnfaithful response:")
    print(f"  {unfaithful_answer[:80]}...")
    result_unfaithful = scorer.score(
        question="Tell me about France",
        answer=unfaithful_answer,
        context=context,
        use_llm=False
    )
    print(f"  Faithfulness score: {result_unfaithful.score:.2f}")
    print(f"  {result_unfaithful.explanation}")

    # Example 2: Citation checking
    print("\n2. Citation Accuracy Check")
    print("-" * 40)

    response_with_citations = """
    Paris is the capital of France [1]. The Eiffel Tower was built in 1889 [2].
    The tower is 324 meters tall [3]. France has 67 million people [1].
    """

    documents = {
        "1": "France is a country with Paris as its capital. Population: 67 million.",
        "2": "The Eiffel Tower was constructed in 1889 for the World's Fair.",
        # "3" is missing - invalid citation
    }

    checker = CitationChecker()
    citation_result = checker.check_citations(response_with_citations, documents)

    print(f"Response: {response_with_citations[:60]}...")
    print(f"\nCitation accuracy: {citation_result['accuracy']:.0%}")
    print(f"Valid citations: {citation_result['valid_citations']}/{citation_result['total_citations']}")
    print(f"Invalid citations: {citation_result['invalid_citations']}")

    # Example 3: Using LLM (shown as pseudo-code)
    print("\n3. LLM-Based Evaluation (requires API key)")
    print("-" * 40)
    print("""
# With LLM enabled:
scorer = FaithfulnessScorer(model="gpt-4o-mini")
result = scorer.score(
    question="What is the capital of France?",
    answer="Paris is the capital of France and is known as the City of Light.",
    context="France is a country in Europe. Paris is its capital city.",
    use_llm=True
)

# The LLM will:
# 1. Extract claims: ["Paris is the capital of France", "Paris is known as the City of Light"]
# 2. Verify each claim against context
# 3. First claim: SUPPORTED (stated in context)
# 4. Second claim: UNVERIFIABLE (not in context, but not contradicted)
""")

    print("\n" + "=" * 60)
    print("Key Takeaway: Faithfulness evaluation checks if the response")
    print("accurately reflects the retrieved context. High faithfulness")
    print("prevents hallucination but may miss information from model's")
    print("parametric knowledge. Balance based on use case requirements.")
    print("=" * 60)
