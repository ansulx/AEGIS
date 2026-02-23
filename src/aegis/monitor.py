"""
Clinical monitor: wraps models or pipelines with guardrails and logging.

Use ClinicalMonitor to run your clinical AI component through AEGIS
guardrails and optional audit logging.
"""

from typing import Any, Callable, Optional

from .guardrails import Guardrail


class ClinicalMonitor:
    """
    Wraps a model or callable with a guardrail and optional logging.

    Example:
        monitor = ClinicalMonitor(guardrail=Guardrail(confidence_threshold=0.9))
        result = monitor.run(my_model, input_data)
    """

    def __init__(
        self,
        guardrail: Optional[Guardrail] = None,
        log_calls: bool = True,
    ):
        self.guardrail = guardrail or Guardrail()
        self.log_calls = log_calls

    def run(
        self,
        model_or_fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Run the model/function with guardrail checks.

        Args:
            model_or_fn: Callable that takes input and returns (output, confidence) or output with .confidence.
            *args, **kwargs: Passed through to model_or_fn.

        Returns:
            Model output if guardrail passes; otherwise behavior depends on fallback_action.
        """
        raw = model_or_fn(*args, **kwargs)
        confidence = getattr(raw, "confidence", None)
        if isinstance(raw, tuple):
            output, confidence = raw[0], raw[1] if len(raw) > 1 else confidence
        else:
            output = raw

        if self.guardrail.check(output, confidence):
            return output
        if self.guardrail.fallback_action == "reject":
            raise ValueError("Guardrail rejected low-confidence output")
        if self.guardrail.fallback_action == "escalate":
            return {"status": "escalate", "raw_output": output, "confidence": confidence}
        return output
