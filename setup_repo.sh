#!/bin/bash
set -euo pipefail

ROOT="project"

mkdir -p "$ROOT/English/checkpoints"
mkdir -p "$ROOT/Telugu/checkpoints"
mkdir -p "$ROOT/Results/English/test_generated_audios"
mkdir -p "$ROOT/Results/English/variance_adaptor_plots/duration"
mkdir -p "$ROOT/Results/English/variance_adaptor_plots/pitch"
mkdir -p "$ROOT/Results/English/variance_adaptor_plots/energy"
mkdir -p "$ROOT/Results/Telugu/test_generated_audios"
mkdir -p "$ROOT/Results/Telugu/variance_adaptor_plots/duration"
mkdir -p "$ROOT/Results/Telugu/variance_adaptor_plots/pitch"
mkdir -p "$ROOT/Results/Telugu/variance_adaptor_plots/energy"
mkdir -p "$ROOT/Report"

# English code/config
touch "$ROOT/English/train.py"
touch "$ROOT/English/preprocess.py"
touch "$ROOT/English/inference.py"
touch "$ROOT/English/config.yaml"
touch "$ROOT/English/checkpoints/.gitkeep"

# Telugu code/config
touch "$ROOT/Telugu/train.py"
touch "$ROOT/Telugu/preprocess.py"
touch "$ROOT/Telugu/inference.py"
touch "$ROOT/Telugu/config.yaml"
touch "$ROOT/Telugu/checkpoints/.gitkeep"

# Result placeholder files
touch "$ROOT/Results/English/training_loss.png"
touch "$ROOT/Results/English/validation_loss.png"
touch "$ROOT/Results/Telugu/training_loss.png"
touch "$ROOT/Results/Telugu/validation_loss.png"

# Root files
touch "$ROOT/README.md"
touch "$ROOT/requirements.txt"
touch "$ROOT/PROJECT_SPEC.md"

# TODO comments in empty source/config files
echo "# TODO: implement training loop" > "$ROOT/English/train.py"
echo "# TODO: implement preprocessing pipeline" > "$ROOT/English/preprocess.py"
echo "# TODO: implement inference script" > "$ROOT/English/inference.py"
echo "# TODO: fill in hyperparameters" > "$ROOT/English/config.yaml"
echo "# TODO: implement training loop" > "$ROOT/Telugu/train.py"
echo "# TODO: implement preprocessing pipeline" > "$ROOT/Telugu/preprocess.py"
echo "# TODO: implement inference script" > "$ROOT/Telugu/inference.py"
echo "# TODO: fill in hyperparameters" > "$ROOT/Telugu/config.yaml"

echo "Repo tree created under $ROOT/"
