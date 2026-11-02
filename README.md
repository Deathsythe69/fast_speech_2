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
│   └── Technical_Report.pdf          # 10-page comprehensive technical report
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

| ID | Lang | Test Description & Phonetic Focus | Duration | RTF (CPU) | Symbols | Frames |
|:---:|:---:|:---|:---:|:---:|:---:|:---:|
| **E1** | EN | Multisyllabic stress test (*antidisestablishmentarianism*) | 10.25 s | 0.38x | 96 | 883 |
| **E2a** | EN | Noun lexical stress on *"conduct"* (`/K AA1 N D AH0 K T/`) | 6.59 s | 0.27x | 61 | 568 |
| **E2b** | EN | Verb lexical stress on *"conduct"* (`/K AH0 N D AH1 K T/`) | 8.58 s | 0.27x | 80 | 739 |
| **E3** | EN | Typo space pause boundary (*"rhyth m"*) | 7.64 s | 0.27x | 71 | 658 |
| **E4** | EN | Numeral expansion (`10:45 AM`, `12 November 2026`) + loanword (*"andaroo"*) | 10.15 s | 0.27x | 95 | 874 |
| **T1** | TE | Pharmaceutical loanword (*న్యూరోఫార్మకోలాజికల్*) | 7.19 s | 0.05x | 71 | 619 |
| **T2a** | TE | Homograph *"పట్టు"* as independent head noun (physical grip) | 4.54 s | 0.02x | 36 | 391 |
| **T2b** | TE | Homograph *"పట్టు"* in compound noun (*"పట్టుచీరను"*, silk saree) | 4.46 s | 0.02x | 41 | 384 |
| **T3** | TE | Concise narrative prosody (*పులి అడవిలో వేగంగా పరిగెత్తింది.*) | 3.85 s | 0.02x | 33 | 332 |
| **T4** | TE | Date/time numeral expansion (`2026`, `10:45`, `12వ`) + conjuncts (*సిద్ధంగా*) | 11.24 s | 0.01x | 130 | 968 |

### Summary of Key Acoustic Findings:
1. **Lexical Stress Differentiation (E2a vs E2b)**: In E2a (noun), the primary stress vowel `/AA1/` shows longer duration and higher pitch peak (182 Hz vs 142 Hz). In E2b (verb), the second syllable `/AH1/` exhibits higher duration (152 ms) and an energy peak (195 Hz), confirming successful POS-guided stress shifting.
2. **Abugida Homograph Analysis (T2a vs T2b)**: In T2a, *"పట్టు"* acts as an independent head noun with focal accent, producing an extended geminate occlusion on `/t_/` (115 ms) and elevated $F_0$ (265 Hz). In T2b, *"పట్టు"* is a modifier in *"పట్టుచీరను"*, where stress shifts to the head compound, reducing geminate duration to 82 ms.
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

The complete academic analysis is compiled in **`Report/Technical_Report.pdf`** (10 pages), covering:
- **Section 1**: Introduction, Objectives, Non-Autoregressive vs Autoregressive Paradigms
- **Section 2**: Architecture Lineage, FFT Blocks, Variance Adaptor, and Audited Bug Fixes Table
- **Section 3**: Datasets (Svarah & IndicTTS), Transcript-Disjoint Splitting, Preprocessing Pipeline, English & Telugu Frontends
- **Section 4**: Training Trajectories & Validation Loss Analysis (Mel, PostNet Mel, Duration, Pitch, Energy)
- **Section 5**: Per-Sentence Synthesis Stats, Variance Adaptor Contour Analysis, and Homograph Stress Variations
- **Section 6**: Comprehensive Test Sentence Analysis (E1–E4, T1–T4) with Embedded Duration, Pitch, and Energy Plots
- **Section 7**: Cross-Lingual Comparison (English Multi-Speaker vs Telugu Single-Speaker), Limitations, and Failure Modes
- **Section 8**: Reproducibility Details, Execution Commands, and Submission Manifest

---

## 7. Checkpoint Verification Manifest

| Language | Checkpoint Path | Size | SHA-256 Hash |
|:---|:---|:---:|:---|
| **English** | `English/checkpoints/best.pth` | 140.5 MB | `84c421d9e03d77cd3f732064b75393fe7abb95f60d06427fa6deb61dd69e5192` |
| **Telugu** | `Telugu/checkpoints/best.pth` | 140.4 MB | `82b39fe49ebe0012837fda23a25ac95eb869bc4271a70c10bc9d40becbe315e3` |

---

## 8. Academic Integrity & AI Disclosure

In accordance with academic integrity guidelines:
- **Author Block**: Debasis Panigrahi, B.Tech in Computer Science and Information Technology, Trident Academy of Technology, Bhubaneswar.
- **AI Coding Assistant**: Google Antigravity (Gemini 2.5 Pro) was utilized for code scaffolding, debugging, and documentation assistance.
- **Execution & Evaluation**: All model training runs (20,000 steps on NVIDIA RTX 4060 GPU), acoustic evaluations, phoneme alignments, and subjective listening analyses were conducted by the author.
