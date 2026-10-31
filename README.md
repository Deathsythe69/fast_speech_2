# FastSpeech 2 Text-to-Speech (English + Telugu)

A production-grade, non-autoregressive Text-to-Speech (TTS) system implementing two separate **FastSpeech 2** acoustic models:
1. **English Model**: Multi-speaker conditioned model trained on the IISc Svarah Indian English dataset (78-symbol ARPAbet inventory).
2. **Telugu Model**: Single-speaker dedicated model trained on the IndicTTS Telugu female voice corpus (58-symbol rule-based phonetic inventory).

Audio output is generated at **22,050 Hz, 16-bit mono PCM WAV** via a pretrained universal **HiFi-GAN** neural vocoder.

---

## 1. Project Structure

```
fast_speech_2/
├── English/                          # Self-contained English model & pipeline
│   ├── checkpoints/                  # Checkpoints (best.pth, last.pth)
│   ├── logs/                         # Training logs (train_losses.csv, val_losses.csv, run_info.json)
│   ├── model/                        # FastSpeech 2 model, transformer blocks, variance adaptor, optimizer
│   ├── text/                         # English text frontend (normalizer, CMUdict + POS heteronym G2P)
│   ├── utils/                        # Audio processing, model helpers, tools
│   ├── config.yaml                   # Unified English configuration
│   ├── train.py                      # Self-contained English training entrypoint
│   ├── inference.py                  # CLI inference script
│   └── CHANGES.md                    # Modifications log relative to ming024/FastSpeech2
│
├── Telugu/                           # Self-contained Telugu model & pipeline
│   ├── checkpoints/                  # Checkpoints (best.pth, last.pth)
│   ├── logs/                         # Training logs (train_losses.csv, val_losses.csv, run_info.json)
│   ├── model/                        # FastSpeech 2 model, transformer blocks, variance adaptor, optimizer
│   ├── text/                         # Telugu text frontend (normalizer, rule-based phonetic G2P)
│   ├── utils/                        # Audio processing, model helpers, tools
│   ├── config.yaml                   # Unified Telugu configuration
│   ├── train.py                      # Self-contained Telugu training entrypoint
│   ├── inference.py                  # CLI inference script
│   └── CHANGES.md                    # Modifications log relative to ming024/FastSpeech2
│
├── Results/                          # Generated evaluation artifacts
│   ├── English/                      # Test audios (.wav, .json), variance plots (.png), val_metrics.md
│   └── Telugu/                       # Test audios (.wav, .json), variance plots (.png), val_metrics.md
│
├── tools/                            # Automation and evaluation utilities
│   ├── select_subset.py              # Transcript-disjoint dataset subsampler
│   ├── align.py                      # MFA alignment generator and validator
│   ├── plot_variance.py              # Variance adaptor visualizer (duration, pitch, energy)
│   ├── sentence_stats.py             # Per-sentence prosody and RTF statistics
│   ├── eval_val.py                   # Quantitative teacher-forced and free-running validation
│   └── run_pipeline.py               # End-to-end pipeline orchestrator
│
├── tests/                            # Unit and integration test suite (20 tests, 100% passing)
├── vocoder/                          # Pretrained HiFi-GAN vocoder config and universal weights
├── test_sentences.json               # Required evaluation test sentences (E1–E4, T1–T4)
├── run_inference.sh                  # One-line cross-platform bash pipeline runner
├── run_inference.ps1                 # One-line Windows PowerShell pipeline runner
├── AUDIT.md                          # Comprehensive bug audit and hardening report (16 findings)
├── CLAIMS_CHECK.md                   # Formal verification of all numbers and claims
├── DEFENSE_NOTES.md                  # Viva voce examination guide (architecture, 9 decisions, 25 Q&As)
├── PROJECT_SPEC.md                   # Engineering project specification and decisions log
└── README.md
```

---

## 2. Environment Setup

### Prerequisites
- Python 3.10 or 3.11
- PyTorch >= 2.0 with CUDA support (or CPU mode for dry-runs and inference)
- SoundFile, Librosa, PyWorld, Matplotlib, PyYAML, TQDM, SciPy, NLTK

### Installation
```bash
# Clone the repository
git clone <repo_url>
cd fast_speech_2

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1

# Install pinned dependencies
pip install -r requirements.txt

# Download required NLTK tagger data for English POS heteronym G2P
python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng'); nltk.download('averaged_perceptron_tagger')"
```

---

## 3. Data Preparation & Preprocessing

The dataset subsampler extracts balanced, transcript-disjoint splits (~5 hours train, ~1 hour val) with audio normalized to 22,050 Hz 16-bit mono PCM within [1.0, 12.0] seconds duration.

```bash
# 1. English Subsampling (Svarah)
python tools/select_subset.py --lang english --src "<SVARAH_ROOT>" --manifest "<SVARAH_CSV>" \
  --col_id id --col_audio audio_filepath --col_text text --col_speaker speaker \
  --train_hours 5 --val_hours 1 --min_dur 1.0 --max_dur 12.0 --seed 1234

# 2. Telugu Subsampling (IndicTTS)
python tools/select_subset.py --lang telugu --src "<INDIC_ROOT>" --manifest "<INDIC_CSV>" \
  --col_id id --col_audio audio_filepath --col_text text --col_speaker speaker \
  --train_hours 5 --val_hours 1 --min_dur 1.0 --max_dur 12.0 --seed 1234

# 3. MFA Alignment & TextGrid Generation
python tools/align.py --lang english
python tools/align.py --lang telugu

# 4. Feature Extraction & No-Leak Normalization
python English/preprocess.py English/config.yaml
python Telugu/preprocess.py Telugu/config.yaml
```

> [!NOTE]
> Preprocessing computes pitch and energy normalization parameters (`stats.json`) strictly from the **train partition only**, completely preventing validation data leakage.

---

## 4. Training

Both `English/train.py` and `Telugu/train.py` are self-contained and support multi-GPU / single-GPU / CPU execution, gradient accumulation, Adam optimizer with Noam warmup and multi-step annealing, NaN/Inf crash dumping, and a built-in `--smoke_test` mode.

```bash
# Standard GPU Training (English)
python English/train.py --config English/config.yaml --device cuda

# Standard GPU Training (Telugu)
python Telugu/train.py --config Telugu/config.yaml --device cuda

# Quick 100-step Smoke Test on CPU (Verification)
python English/train.py --smoke_test --max_steps 100 --device cpu
python Telugu/train.py --smoke_test --max_steps 100 --device cpu

# Resume from Checkpoint
python English/train.py --resume English/checkpoints/last.pth
```

### Training Outputs:
- **Checkpoints**: Saved to `<Lang>/checkpoints/`:
  * `best.pth`: Lowest validation loss checkpoint
  * `last.pth`: Most recent step checkpoint
  * `step_<step>.pth`: Checkpoints saved at periodic intervals
- **Loss Logs**:
  * `<Lang>/logs/train_losses.csv`: Step, epoch, lr, total, mel, postnet_mel, pitch, energy, duration, grad_norm, step_duration
  * `<Lang>/logs/val_losses.csv`: Evaluated on full validation set
  * `<Lang>/logs/run_info.json`: Hardware, seeds, git hash, configuration SHA256

---

## 5. Inference & Evaluation

### One-Line End-to-End Runner

Executes synthesis, variance plotting, per-sentence statistics, and quantitative validation in sequence:

```bash
# Linux / macOS (Bash)
./run_inference.sh english English/checkpoints/best.pth
./run_inference.sh telugu Telugu/checkpoints/best.pth
./run_inference.sh both

# Windows (PowerShell)
powershell -ExecutionPolicy Bypass -File run_inference.ps1 english English/checkpoints/best.pth
powershell -ExecutionPolicy Bypass -File run_inference.ps1 telugu Telugu/checkpoints/best.pth
powershell -ExecutionPolicy Bypass -File run_inference.ps1 both
```

### Standalone Inference Commands

```bash
# English Synthesis
python English/inference.py \
  --checkpoint English/checkpoints/best.pth \
  --out_dir Results/English \
  --device cpu \
  --pitch_control 1.0 \
  --energy_control 1.0 \
  --duration_control 1.0

# Telugu Synthesis
python Telugu/inference.py \
  --checkpoint Telugu/checkpoints/best.pth \
  --out_dir Results/Telugu \
  --device cpu \
  --pitch_control 1.0 \
  --energy_control 1.0 \
  --duration_control 1.0
```

### Variance Adaptor Plotting
```bash
python tools/plot_variance.py --lang english --out_dir Results/English
python tools/plot_variance.py --lang telugu --out_dir Results/Telugu
```

---

## 6. Summary of Test Sentences & Results

All 10 required evaluation sentences from `test_sentences.json` were synthesized and evaluated.

| ID | Language | Description / Test Focus | Duration (s) | RTF (CPU) |
|:---:|:---:|:---|:---:|:---:|
| **E1** | English | Multisyllabic stress test (*antidisestablishmentarianism*) | 1.13 s | 0.35x |
| **E2a** | English | Noun stress on *"conduct"* (`/K AA1 N D AH0 K T/`) | 0.71 s | 0.32x |
| **E2b** | English | Verb stress on *"conduct"* (`/K AH0 N D AH1 K T/`) | 0.77 s | 0.31x |
| **E3** | English | Typo space handling (*"rhyth m"*) | 0.51 s | 0.34x |
| **E4** | English | Numeral expansion (`10:45 AM`, `12 November 2026`) + OOV (*"andaroo"*) | 1.16 s | 0.29x |
| **T1** | Telugu | Multisyllabic phonetic loanword (*న్యూరోఫార్మకోలాజికల్*) | 1.01 s | 0.28x |
| **T2a** | Telugu | Homograph *"పట్టు"* (physical grip) | 0.55 s | 0.29x |
| **T2b** | Telugu | Homograph *"పట్టు"* (silk saree compound) | 0.56 s | 0.28x |
| **T3** | Telugu | Animal / narrative prosody (*పులి అడవిలో...*) | 0.49 s | 0.32x |
| **T4** | Telugu | Date/time numeral expansion (`2026`, `10:45`, `12వ`) + conjuncts | 1.80 s | 0.28x |

### Specialized Comparison Plots
- `Results/English/variance_adaptor_plots/comparisons/conduct_E2a_vs_E2b.png`: Demonstrates noun vs verb stress shift across duration, pitch ($F_0$), and energy on the target root `/K AA1 N D AH0 K T/` vs `/K AH0 N D AH1 K T/`.
- `Results/Telugu/variance_adaptor_plots/comparisons/pattu_T2a_vs_T2b.png`: Compares prosodic prominence between independent head noun *"పట్టు"* (T2a) and compound modifier *"పట్టుచీరను"* (T2b).

---

## 7. Automated Test Suite

A comprehensive pytest suite verifies all components across frontend, audio DSP, inference, and plotting:

```bash
python -m pytest tests/ -v
```

**Results**: `20 passed, 0 failed (100% pass rate)`.
- `tests/test_frontend_en.py`: Dates, times, apostrophe preservation, conduct heteronym distinction, OOV fallback (5 tests).
- `tests/test_frontend_te.py`: Numeral expansion, ZWJ/ZWNJ handling, homograph 'పట్టు', full symbol inventory (4 tests).
- `tests/test_inference.py`: 22050 Hz 16-bit PCM WAV checks, variance plot existence, comparison figures, NPZ consistency, pipeline manifests (9 tests).
- `tests/test_preprocess.py`: Feature extraction dimensions, no-leak train-only normalization verification (2 tests).

---

## 8. AI Attribution Statement

In accordance with academic integrity guidelines:
- **Primary AI Assistant**: Google Antigravity / Gemini 2.5 Pro Agentic Pair Programming System.
- **Tasks Assisted by AI**:
  * Automated scaffolding and file partitioning into modular `English/` and `Telugu/` directories.
  * Rapid regex drafting for Telugu and English numeral, ordinal, date, and time text normalizers.
  * Comprehensive test suite construction (`tests/`) and pipeline orchestration scripts (`tools/run_pipeline.py`).
- **Human Verification & Critical Review**:
  * Formulated and verified all phonetic mapping rules, virama (`్`) suppression logic, and conjunct decompositions.
  * Audited and corrected reference repository bugs: variance control coupling (`modules.py`), Length Regulator empty tensor collapse, and ScheduledOptim state loss upon resume.
  * Audited data splits for zero train/val overlap and ensured non-leaking statistics computation.
  * Executed all validation benchmarks, auditory verification of synthesized WAV files, and inspection of variance plots.
