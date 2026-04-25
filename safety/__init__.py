"""
Chapter 13: Safety Evaluation (Red-Teaming)
===========================================

Evaluate AI systems for safety vulnerabilities:
- Prompt injection detection
- Jailbreak resistance
- Attack pattern recognition
- Automated red-teaming

Book Reference: Chapter 13 covers adversarial evaluation -
"Standard evaluation tests whether systems work as intended.
Red-teaming tests whether systems can be made to fail."
"""

from .red_team_patterns import (
    AttackPattern,
    AttackDetector,
    generate_attack_variations,
)
from .prompt_injection import (
    InjectionDetector,
    test_injection_resistance,
)

__all__ = [
    "AttackPattern",
    "AttackDetector",
    "generate_attack_variations",
    "InjectionDetector",
    "test_injection_resistance",
]
