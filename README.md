# FastSpeech 2 TTS Project

## Overview
## Environment
## Data preparation
## Preprocessing
## Training
## Inference

FastSpeech 2 inference is self-contained and config-driven per language. It synthesizes audio from checkpoints, extracts phoneme-level and frame-level variance arrays (`duration`, `pitch`, `energy`, `mel`), generates variance adaptor plots, computes per-sentence statistics, and evaluates quantitative validation metrics.

### 1. Single Command Pipeline Execution

**Linux / macOS (Bash):**
```bash
# English inference & analysis
./run_inference.sh english checkpoints/english/last.pth

# Telugu inference & analysis
./run_inference.sh telugu checkpoints/telugu/last.pth
```

**Windows (PowerShell):**
```powershell
# English inference & analysis
powershell -ExecutionPolicy Bypass -File run_inference.ps1 english checkpoints/english/last.pth

# Telugu inference & analysis
powershell -ExecutionPolicy Bypass -File run_inference.ps1 telugu checkpoints/telugu/last.pth
```

### 2. Individual Component Execution

**Synthesis (English):**
```bash
python English/inference.py \
  --checkpoint checkpoints/english/last.pth \
  --out_dir Results/English \
  --device cpu \
  --speaker speaker_0 \
  --pitch_control 1.0 \
  --energy_control 1.0 \
  --duration_control 1.0
```

**Synthesis (Telugu):**
```bash
python Telugu/inference.py \
  --checkpoint checkpoints/telugu/last.pth \
  --out_dir Results/Telugu \
  --device cpu \
  --pitch_control 1.0 \
  --energy_control 1.0 \
  --duration_control 1.0
```

**Variance Adaptor Plotting:**
```bash
python tools/plot_variance.py --lang english --results_dir Results
python tools/plot_variance.py --lang telugu --results_dir Results
```

**Per-Sentence Variance Statistics:**
```bash
python tools/sentence_stats.py --lang both --results_dir Results
```

**Quantitative Validation Metrics:**
```bash
python tools/eval_val.py --lang english --checkpoint checkpoints/english/last.pth --device cpu
python tools/eval_val.py --lang telugu --checkpoint checkpoints/telugu/last.pth --device cpu
```
## Results
## Checkpoints
## Reproducibility commands
