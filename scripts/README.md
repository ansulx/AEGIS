# Scripts

Utility scripts for evaluation, benchmarking, and reproducibility will go here.

## Available scripts

- `python scripts/check_data_loading.py`
  - Validates `adam data/` and `Data_Set/` structure and pairing.
- `python scripts/run_research_pipeline.py`
  - Strict policy pipeline:
    - train on ADAM only
    - inference on ADAM + Indian cohorts
    - Monte Carlo uncertainty outputs
    - CUDA GPU is required by default (`--allow-cpu` to override)
