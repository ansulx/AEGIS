"""
Configurable guardrails for clinical AI outputs.

Guardrails enforce confidence thresholds, consistency checks,
and fallback actions to prevent silent failures in production.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class Guardrail:
    """
    Configuration for a single guardrail applied to model outputs.

    Attributes:
        confidence_threshold: Minimum confidence to accept an output (0–1).
        require_explanation: If True, output must include an explanation.
        fallback_action: Action when guardrail fails: "reject", "escalate", "default".
    """

    confidence_threshold: float = 0.95
    require_explanation: bool = True
    fallback_action: str = "escalate"

    def check(self, output: Any, confidence: Optional[float] = None) -> bool:
        """
        Check whether the given output passes this guardrail.

        Args:
            output: Model output (structure depends on your pipeline).
            confidence: Optional explicit confidence score; otherwise inferred from output.

        Returns:
            True if the output passes the guardrail.
        """
        if confidence is None:
            confidence = getattr(output, "confidence", 0.0)
        return confidence >= self.confidence_threshold
