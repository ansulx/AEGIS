# AEGIS Quick Start

This guide walks you through using AEGIS guardrails with a minimal example.

## Install

```bash
pip install -r requirements.txt
# or: pip install -e .
```

## Basic usage

```python
from aegis import Guardrail, ClinicalMonitor

# 1. Define a guardrail
guardrail = Guardrail(
    confidence_threshold=0.95,
    require_explanation=True,
    fallback_action="escalate",
)

# 2. Create a monitor
monitor = ClinicalMonitor(guardrail=guardrail)

# 3. Wrap your model (example: a function returning output and confidence)
def my_clinical_model(x):
    # Your model here
    return {"prediction": 1, "confidence": 0.97}

result = monitor.run(my_clinical_model, some_input)
```

## Next steps

- See `examples/` for full pipelines.
- Read the main [README](../README.md) for citation and project structure.
