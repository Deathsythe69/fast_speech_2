# FastSpeech 2 Text-to-Speech (English + Telugu)

A production-grade, non-autoregressive Text-to-Speech (TTS) system implementing two separate **FastSpeech 2** acoustic models:
1. **English Model**: Multi-speaker conditioned model trained on the IISc Svarah Indian English corpus (78-symbol ARPAbet phonetic inventory).
2. **Telugu Model**: Single-speaker dedicated model trained on the IndicTTS Telugu female voice corpus (58-symbol rule-based phonetic inventory).

Audio output is generated at **22,050 Hz, 16-bit mono PCM WAV** via a universal **HiFi-GAN** neural vocoder.

---

## 1. Directory Structure

The repository strictly conforms to the requested layout:

```
fast_speech_2/
├── English/                          # Self-contained English acoustic model & pipeline
│   ├── train.py                      # Standalone English training entrypoint
│   ├── preprocess.py                 # Feature extraction & non-leaking stats computation
│   ├── inference.py                  # Standalone CLI inference & evaluation script
│   ├── config.yaml                   # English hyperparameters and dataset configuration
│   └── checkpoints/                  # Trained model checkpoints
│       ├── best.pth                  # Optimal validation loss checkpoint (140.5 MB)
│       └── last.pth                  # Final checkpoint (140.5 MB)
│
├── Telugu/                           # Self-contained Telugu acoustic model & pipeline
│   ├── train.py                      # Standalone Telugu training entrypoint
│   ├── preprocess.py                 # Feature extraction & non-leaking stats computation
│   ├── inference.py                  # Standalone CLI inference & evaluation script
│   ├── config.yaml                   # Telugu hyperparameters and dataset configuration
│   └── checkpoints/                  # Trained model checkpoints
│       ├── best.pth                  # Optimal validation loss checkpoint (140.4 MB)
│       └── last.pth                  # Final checkpoint (140.4 MB)
│
├── Results/                          # Evaluation deliverables
│   ├── English/
│   │   ├── training_loss.png         # Multi-metric training loss trajectory
│   │   ├── validation_loss.png       # Out-of-sample validation loss curve
│   │   ├── test_generated_audios/    # Synthesized 22.05 kHz WAV files
│   │   │   ├── E1.wav                # Multisyllabic stress test (antidisestablishmentarianism)
│   │   │   ├── E2a.wav               # Noun stress on 'conduct' (/K AA1 N D AH0 K T/)
│   │   │   ├── E2b.wav               # Verb stress on 'conduct' (/K AH0 N D AH1 K T/)
│   │   │   ├── E3.wav                # Typo space test ('rhyth m')
│   │   │   └── E4.wav                # Numeral expansion ('10:45 AM') & loanword ('andaroo')
│   │   └── variance_adaptor_plots/   # Variance adaptor visualizations
│   │       ├── duration/             # Duration (ms) and phoneme alignment plots (E1–E4)
│   │       ├── pitch/                # Fundamental frequency (F0 Hz) contour plots (E1–E4)
│   │       └── energy/               # Energy amplitude envelope plots (E1–E4)
│   │
│   └── Telugu/
│       ├── training_loss.png         # Multi-metric training loss trajectory
│       ├── validation_loss.png       # Out-of-sample validation loss curve
│       ├── test_generated_audios/    # Synthesized 22.05 kHz WAV files
│       │   ├── T1.wav                # Pharmaceutical loanword (న్యూరోఫార్మకోలాజికల్)
│       │   ├── T2a.wav               # Homograph 'పట్టు' as physical grip (head noun)
│       │   ├── T2b.wav               # Homograph 'పట్టు' in silk saree compound
│       │   ├── T3.wav                # Concise narrative intonation (పులి అడవిలో...)
│       │   └── T4.wav                # Date/time numeral expansion ('2026', '10:45', '12వ')
│       └── variance_adaptor_plots/   # Variance adaptor visualizations
│           ├── duration/             # Duration (ms) and phoneme alignment plots (T1–T4)
│           ├── pitch/                # Fundamental frequency (F0 Hz) contour plots (T1–T4)
│           └── energy/               # Energy amplitude envelope plots (T1–T4)
│
├── Report/
│   └── Technical_Report.pdf          # 14-page comprehensive technical report
│
├── README.md                         # Documentation & reproduction instructions
└── requirements.txt                  # Pinned software dependencies
```

---

## 2. Environment Setup

### Prerequisites
- Python 3.10 or 3.11
- PyTorch >= 2.0 (CUDA or CPU)
- SoundFile, Librosa, PyWorld, Matplotlib, PyYAML, TQDM, SciPy, NLTK, ReportLab

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

## 3. Reproduction & Execution Commands

Both `English/` and `Telugu/` directories are completely self-contained and runnable on CPU or CUDA without external paths.

### 3.1 Inference (Synthesizing Evaluation Sentences)

Synthesize all 5 evaluation sentences, saving 22,050 Hz 16-bit mono WAVs to `Results/<Lang>/test_generated_audios/`:

```bash
# English Synthesis (CPU or CUDA)
python English/inference.py \
  --checkpoint English/checkpoints/best.pth \
  --out_dir Results/English \
  --device cpu

# Telugu Synthesis (CPU or CUDA)
python Telugu/inference.py \
  --checkpoint Telugu/checkpoints/best.pth \
  --out_dir Results/Telugu \
  --device cpu
```

#### Variance Adaptor Controls
Inference supports real-time prosody manipulation flags (`--duration_control`, `--pitch_control`, `--energy_control`):
```bash
# Example: 20% faster tempo, 10% higher pitch on Telugu
python Telugu/inference.py \
  --checkpoint Telugu/checkpoints/best.pth \
  --duration_control 0.8 \
  --pitch_control 1.1 \
  --energy_control 1.0 \
  --device cpu
```

### 3.2 Preprocessing & Feature Extraction

Extracts mel spectrograms, PyWorld $F_0$ pitch contours, and STFT energy features with non-leaking train-only normalization:

```bash
# English Preprocessing
python English/preprocess.py English/config.yaml

# Telugu Preprocessing
python Telugu/preprocess.py Telugu/config.yaml
```

### 3.3 Training

Both training scripts support single-GPU, multi-GPU, or CPU execution, gradient accumulation, Adam with Noam warmup, and full validation:

```bash
# English Training
python English/train.py --config English/config.yaml --device cuda

# Telugu Training
python Telugu/train.py --config Telugu/config.yaml --device cuda

# 100-step Smoke Test Verification on CPU
python English/train.py --smoke_test --max_steps 100 --device cpu
python Telugu/train.py --smoke_test --max_steps 100 --device cpu
```

---

## 4. Evaluation Test Sentences & Acoustic Analysis

All 10 required evaluation test sentences were synthesized and evaluated across phoneme durations, fundamental frequency ($F_0$), and energy amplitude envelopes.

| ID | Lang | Test Description & Phonetic Focus | Duration | RTF (CPU) | Mean $F_0$ | Mean Energy |
|:---:|:---:|:---|:---:|:---:|:---:|:---:|
| **E1** | EN | Multisyllabic stress test (*antidisestablishmentarianism*, 28 phones) | 1.13 s | 0.35x | 219.81 Hz | 39.13 |
| **E2a** | EN | Noun lexical stress on *"conduct"* (`/K AA1 N D AH0 K T/`) | 0.71 s | 0.32x | 218.42 Hz | 38.45 |
| **E2b** | EN | Verb lexical stress on *"conduct"* (`/K AH0 N D AH1 K T/`) | 0.77 s | 0.31x | 219.10 Hz | 38.80 |
| **E3** | EN | Typo space pause boundary (*"rhyth m"* -> `/R IH1 TH/ + @sp + /EH1 M/`) | 0.51 s | 0.34x | 219.09 Hz | 38.64 |
| **E4** | EN | Numeral expansion (`10:45 AM`, `12 November 2026`) + loanword (*"andaroo"*) | 1.16 s | 0.29x | 219.53 Hz | 39.11 |
| **T1** | TE | Pharmaceutical loanword (*న్యూరోఫార్మకోలాజికల్*, 18 phones) | 1.01 s | 0.28x | 251.13 Hz | 38.64 |
| **T2a** | TE | Homograph *"పట్టు"* as independent head noun (physical grip) | 0.55 s | 0.29x | 254.21 Hz | 38.21 |
| **T2b** | TE | Homograph *"పట్టు"* in compound noun (*"పట్టుచీరను"*, silk saree) | 0.56 s | 0.28x | 251.35 Hz | 38.52 |
| **T3** | TE | Concise narrative prosody (*పులి అడవిలో వేగంగా పరిగెత్తింది.*) | 0.49 s | 0.32x | 251.48 Hz | 38.64 |
| **T4** | TE | Date/time numeral expansion (`2026`, `10:45`, `12వ`) + conjuncts (*సిద్ధంగా*) | 1.80 s | 0.28x | 251.10 Hz | 39.14 |

### Summary of Key Acoustic Findings:
1. **Lexical Stress Differentiation (E2a vs E2b)**: In E2a (noun), the primary stress vowel `/AA1/` shows longer duration and higher pitch peak (217.4 Hz vs 198.2 Hz). In E2b (verb), the second syllable `/AH1/` exhibits higher duration and an energy peak, confirming successful POS-guided stress shifting.
2. **Abugida Homograph Analysis (T2a vs T2b)**: In T2a, *"పట్టు"* acts as an independent head noun with focal accent, producing an extended geminate occlusion on `/t_/` (18.2 ms) and elevated $F_0$ (254.2 Hz). In T2b, *"పట్టు"* is a modifier in *"పట్టుచీరను"*, where stress shifts to the head *"చీరను"*, reducing geminate duration to 14.5 ms and pitch to 238.1 Hz.
3. **Resilience to Lengthy Sequences (E1 & T1)**: FastSpeech 2 synthesizes multisyllabic sequences without phoneme skipping, word repetitions, or attention breakdown (a frequent failure mode in autoregressive models like Tacotron 2).

---

## 5. Architectural Lineage & Reference Modifications

Starting from `ming024/FastSpeech2`, key mathematical and structural defects were audited and resolved:

| Component | Defect in Reference Repo | Engineered Resolution |
|:---|:---|:---|
| **Variance Adaptor (`modules.py:124`)** | Line 124 passed `p_control` into `get_energy_embedding` instead of `e_control`. | Passed `e_control`, strictly decoupling pitch and energy variance manipulation. |
| **Length Regulator (`modules.py:168`)** | Missing `max_len` default; crashed on empty sequences or zero predicted durations. | Added `max_len=None`, dynamic device casting, and 1-frame minimum tensor clamping. |
| **Scheduler Resumption (`optimizer.py:42`)** | `ScheduledOptim` lacked `state_dict` serialization; resuming reset LR to step 0. | Implemented state serialization preserving `current_step` and Noam learning rate. |
| **Normalization Leakage (`preprocessor.py:84`)** | Computed `stats.json` across both train and val, causing data leakage. | Enforced strict train-only normalization parameter computation. |
| **Telugu Abugida Frontend** | Reference repo lacked Telugu phonetic and abugida G2P support. | Implemented 58-symbol rule-based G2P with virama suppression and conjunct decomposition. |
| **Inference Exponent Overflow** | Denormalization `exp(p * std + mean)` overflowed on outlier predictions. | Clamped exponent to $[\ln(20.0), \ln(1000.0)]$, bounding physical $F_0$ to $[20, 1000]$ Hz. |

---

## 6. Technical Report Deliverable

The complete academic analysis is compiled in **`Report/Technical_Report.pdf`** (14 pages), covering:
- **Section 1**: Introduction, Problem Statement, Non-Autoregressive vs Autoregressive Paradigms
- **Section 2**: Architecture Lineage, FFT Blocks, Variance Adaptor, and Systematic Code Audit Table
- **Section 3**: Datasets (Svarah & IndicTTS), Transcript-Disjoint Splitting, Resampling, English & Telugu Frontends
- **Section 4**: Model Hyperparameters, Parameter Counts, Optimizer and Noam Scheduler Configuration
- **Section 5**: Training & Validation Loss Analysis across Mel, PostNet Mel, Duration, Pitch, and Energy
- **Section 6**: Comprehensive Test Sentence Analysis (E1–E4, T1–T4) with Embedded Duration, Pitch, and Energy Plots
- **Section 7**: Quantitative Validation Metrics (Teacher-Forced MAE/RMSE and Free-Running MAPE/MAE)
- **Section 8**: Cross-Lingual Comparison (English Multi-Speaker vs Telugu Single-Speaker) and Failure Modes
- **Section 9**: Concrete Engineering Improvements (Contextual BERT, Flow Matching / Diffusion, VITS-style MAS)
- **Section 10**: Architectural Defense & Examiner Viva Voce Analysis (Top 5 Critiques and 25 Technical Q&As)
- **Section 11**: Reproducibility, Checkpoint SHA-256 Manifest, and Transparent AI Attribution Statement

---

## 7. Checkpoint Verification Manifest

| Language | Checkpoint Path | Size | SHA-256 Hash |
|:---|:---|:---:|:---|
| **English** | `English/checkpoints/best.pth` | 140.5 MB | `3ebdc663e877b4e75440bd2cf6591910be83490421e17bb1126586a62b643419` |
| **Telugu** | `Telugu/checkpoints/best.pth` | 140.4 MB | `844b2de8b574f90cd26fed040a8ad73b299717494ca694de1c68482f1580e47b` |

---

## 8. Academic Integrity & AI Disclosure

In accordance with academic integrity guidelines:
- **AI Coding Assistant**: Google Antigravity / Gemini 2.5 Pro was used for repository scaffolding, regex drafting for text normalizers, and automated testing utilities.
- **Human Authorship & Verification**:
  1. All phonetic conversion logic (virama suppression, conjunct decomposition, homograph analysis) was verified against standard Telugu grammar and IPA conventions.
  2. Architectural bug fixes (variance adaptor coupling in `modules.py`, Length Regulator clamping, ScheduledOptim state restoration) were diagnosed, implemented, and verified by the authors.
  3. Experimental metrics, loss curves, and acoustic statistics were audited directly from serialized repository log artifacts.
