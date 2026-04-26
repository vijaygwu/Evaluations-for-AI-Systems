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

__all__ = [
    "AttackPattern",
    "AttackDetector",
    "generate_attack_variations",
    "InjectionDetector",
    "test_injection_resistance",
]

_EXPORTS = {
    "AttackPattern": ".red_team_patterns",
    "AttackDetector": ".red_team_patterns",
    "generate_attack_variations": ".red_team_patterns",
    "InjectionDetector": ".prompt_injection",
    "test_injection_resistance": ".prompt_injection",
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    module = import_module(_EXPORTS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
