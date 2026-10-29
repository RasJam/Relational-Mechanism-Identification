# Relational Mechanism Identification

Source code and reproducibility files for the MSc project on identifying hidden
2D mechanism constraints from passive observations and controlled active probes.

## Setup

Python 3.12 is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

Use the compact verification before running the complete experiment:

```bash
python run_quick.py
python run_all.py
```

Run the tests with:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The full pipeline generates and validates the simulated dataset, constructs the
absolute and relational representations, trains the classical and neural models,
evaluates OOD performance, and creates the result files and visualisations in
`artifacts/`.

The included dataset and compact reference results are provided for convenient
verification. They can be regenerated from source by running `python run_all.py`.

## Files

- `run_quick.py`: compact end-to-end check.
- `run_all.py`: complete experiment entry point.
- `src/config.py`: experiment settings and feature definitions.
- `src/simulator.py`: mechanism construction, probes, and measurements.
- `src/generate_dataset.py`: dataset generation.
- `src/validate_dataset.py`: integrity and physical-behaviour checks.
- `src/build_features.py`: trajectory summaries for classical models.
- `src/evaluate_baselines.py`: logistic-regression and random-forest evaluation.
- `src/train_encoder_decoder.py`: predictive representation training.
- `src/analyze_neural_slices.py`: OOD analysis by nuisance factor.
- `src/analyze_results.py`: result aggregation.
- `src/create_visuals.py`: figures and animations.
- `tools/create_report_figures.py`: report-specific diagrams.
- `tests/test_pipeline.py`: automated tests.
- `data/`: included simulated dataset.
- `reference_results/`: compact CSV and JSON results from the reported run.
