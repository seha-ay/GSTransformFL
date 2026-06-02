# Model Directory

Place your federated learning model files here before deployment.

## What belongs here

- Model architecture definition file (e.g. `model.py`)
- Pre-trained weights if using transfer learning (e.g. `weights.pt`)
- Any model-specific configuration files

## What does NOT belong here

- Training data — place in `../data/input/`
- NVFlare job configs — those live in `../config/`

## Notes

- GSTransformFL is model-agnostic. It transforms your input data before
  training begins and saves the result to `../data/output/`.
- The downstream training job reads from `../data/output/` — it is
  responsible for loading the model and running training.
- This directory is a placeholder. The model integration is outside
  the scope of GSTransformFL.