"""
AEGIS: A protective shield against silent AI failures in clinical use.

This package provides guardrails, monitoring, and evaluation utilities
for clinical AI systems to detect and mitigate silent failures.
"""

__version__ = "0.1.0"

from .guardrails import Guardrail
from .monitor import ClinicalMonitor

__all__ = ["__version__", "Guardrail", "ClinicalMonitor"]
