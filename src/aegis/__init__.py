"""
AEGIS: A protective shield against silent AI failures in clinical use.

This package provides guardrails, monitoring, and evaluation utilities
for clinical AI systems to detect and mitigate silent failures.
"""

__version__ = "0.1.0"

from .data_loading import DatasetBundle, DatasetLoadError, SessionPair, load_dataset_bundle
from .guardrails import Guardrail
from .monitor import ClinicalMonitor
from .research_pipeline import PipelineConfig, run_research_pipeline

__all__ = [
    "__version__",
    "Guardrail",
    "ClinicalMonitor",
    "DatasetBundle",
    "DatasetLoadError",
    "SessionPair",
    "load_dataset_bundle",
    "PipelineConfig",
    "run_research_pipeline",
]
