# AEGIS

<p align="center">
  <strong>A protective shield against silent AI failures in clinical use</strong>
</p>

<p align="center">
  <a href="https://github.com/ansulx/AEGIS/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"/>
  </a>
  <a href="https://github.com/ansulx/AEGIS">
    <img src="https://img.shields.io/github/stars/ansulx/AEGIS?style=social" alt="GitHub stars"/>
  </a>
  <a href="https://github.com/ansulx/AEGIS/fork">
    <img src="https://img.shields.io/badge/contributions-welcome-brightgreen.svg" alt="Contributions welcome"/>
  </a>
  <a href="https://github.com/ansulx/AEGIS/issues">
    <img src="https://img.shields.io/github/issues/ansulx/AEGIS" alt="GitHub issues"/>
  </a>
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+"/>
  </a>
  <a href="https://github.com/psf/black">
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black"/>
  </a>
</p>

---

## Abstract

**AEGIS** (Adaptive Evaluation and Guardrails for Intelligent Systems) is a research framework and toolkit designed to detect, monitor, and mitigate *silent failures* of AI systems in clinical settings—where models may produce confident but incorrect or unsafe outputs without raising explicit errors. This work addresses the critical need for reliability and interpretability of AI-assisted clinical decision support.

Key objectives:

- **Detection**: Identify when model outputs are unreliable or out-of-distribution in a clinical context.
- **Guardrails**: Provide configurable checks and fallbacks to prevent silent failures from affecting patient care.
- **Auditability**: Enable logging, tracing, and evaluation for regulatory and research compliance.

This repository supports reproducibility for methods and experiments presented in our work (e.g., MICCAI 2026 and related submissions).

---

## Features

- **Silent-failure detection** for clinical AI pipelines  
- **Configurable guardrails** (confidence thresholds, consistency checks, human-in-the-loop triggers)  
- **Evaluation benchmarks** and metrics for clinical reliability  
- **Modular design** for integration with existing clinical AI workflows  
- **Documentation** and examples for research and deployment  

---

## Installation

```bash
# Clone the repository
git clone https://github.com/ansulx/AEGIS.git
cd AEGIS

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode (optional)
pip install -e .
```

---

## Quick Start

```python
from aegis import Guardrail, ClinicalMonitor

# Configure a guardrail for a clinical AI component
guardrail = Guardrail(
    confidence_threshold=0.95,
    require_explanation=True,
    fallback_action="escalate",
)

# Wrap your model or pipeline
monitor = ClinicalMonitor(guardrail=guardrail)
safe_output = monitor.run(your_model, input_data)
```

See [`docs/quickstart.md`](docs/quickstart.md) and [`examples/`](examples/) for detailed tutorials.

---

## MICCAI 2026 Full Run (Strict)

```bash
export PYTHONPATH=src
python src/main.py full_run \
  --baseline-config configs/default.yaml \
  --main-config configs/phase2_swinunetr.yaml \
  --run-name miccai2026_full
```

### What this does

1. **Train** SwinUNETR on ADAM dataset with MC Dropout, early stopping, and mixed precision
2. **Infer** on ADAM (train + holdout) and Indian (CAR + NCAR) datasets with MC Dropout uncertainty
3. **Generate** per-case metrics CSV, NIfTI predictions + uncertainty maps, qualitative NPZ slices
4. **Produce** publication-grade figures (cohort summaries, uncertainty correlation, best/worst gallery)
5. **Run** failure-detection analysis (ROC/AUROC, calibration, optimal flagging threshold)

### CLI Options

| Flag | Description |
|---|---|
| `--allow-cpu` | Allow CPU fallback (default: strict GPU) |
| `--force` | Overwrite existing output directory |
| `--resume-from PATH` | Resume training from checkpoint |

### Outputs

```
outputs/<run-name>/
├── checkpoints/          # Model checkpoints (best, latest, periodic)
├── predictions/          # NIfTI predicted masks by cohort
├── uncertainty/          # NIfTI entropy/uncertainty maps by cohort
├── qualitative/          # NPZ slices for visualization
├── figures/              # Publication-grade PNG figures
│   ├── panels/           # Per-case panels
│   ├── cohort_summary.png
│   ├── uncertainty_correlation.png
│   └── best_worst_gallery.png
├── failure_analysis/     # Failure-detection reports
│   ├── failure_detection_report.json
│   ├── roc_curve.png
│   └── calibration.png
└── reports/
    ├── summary.json
    ├── per_case_metrics.csv
    └── resolved_config.json
```

---

## Project Structure

```
AEGIS/
├── README.md
├── LICENSE
├── requirements.txt
├── configs/
│   ├── default.yaml            # Baseline config (locked parameters)
│   └── phase2_swinunetr.yaml   # Experiment-specific config
├── src/
│   ├── main.py                 # CLI entrypoint
│   └── aegis/
│       ├── __init__.py
│       ├── data_loading.py     # Dataset loading + validation
│       ├── research_pipeline.py # Main pipeline orchestrator
│       ├── training.py         # Training loop (checkpointing, OOM-safe)
│       ├── checkpoint.py       # Atomic checkpoint save/load/resume
│       ├── mc_inference.py     # MC Dropout inference engine
│       ├── visualization.py    # Publication-grade figure generation
│       ├── failure_detection.py # ROC/AUROC, calibration, failure analysis
│       ├── guardrails.py
│       ├── monitor.py
│       ├── models/
│       │   ├── __init__.py
│       │   └── swin_unetr.py  # SwinUNETR with MC Dropout
│       └── evaluation/
├── tests/
├── scripts/
├── examples/
└── docs/
```

---

## Citation

If you use AEGIS in your research, please cite:

```bibtex
@software{aegis2026,
  author       = {Your Name and Collaborators},
  title        = {{AEGIS}: A Protective Shield Against Silent AI Failures in Clinical Use},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/ansulx/AEGIS},
  note         = {MICCAI 2026}
}
```

A `CITATION.cff` file is included for automated citation support (e.g., GitHub and Zenodo).

---

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) and our [Code of Conduct](CODE_OF_CONDUCT.md) before submitting issues or pull requests.

---

## License

This project is licensed under the MIT License—see [LICENSE](LICENSE) for details.

---

## Acknowledgments

This work is developed in the context of clinical AI safety and MICCAI 2026. We thank the medical AI and open-source communities for feedback and support.

Repository: [ansulx/AEGIS](https://github.com/ansulx/AEGIS)
