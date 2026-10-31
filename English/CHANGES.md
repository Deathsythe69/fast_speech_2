# Changes Log — English Model

Tracking all modifications from ming024/FastSpeech2 reference repository.

## Setup
- Copied core architecture: `model/`, `text/`, `audio/`, `utils/`, `transformer/`, `preprocessor/`, `hifigan/` (configs and code only, excluding large weight archives).
- Copied entry scripts: `dataset.py`, `preprocess.py`, `evaluate.py`, `synthesize.py`.
- Removed unused dataset-specific files: `preprocessor/aishell3.py`, `preprocessor/libritts.py`, `config/AISHELL3`, `config/LibriTTS`.

## Text Frontend (`English/text/`)
- Created `English/text/normalizer.py`:
  * Lowercase-safe expansion for times (e.g. `10:45 AM` -> `ten forty-five A M`), dates (e.g. `12 November 2026` -> `twelve November twenty twenty-six`), 4-digit years (`2026` -> `twenty twenty-six`), ordinals, currency, and abbreviations (`Mr.` -> `mister`, etc.).
  * Preserved apostrophes for possessives and contractions (e.g. `orchestra's`).
- Created `English/text/g2p.py`:
  * CMUdict + g2p_en with POS-based heteronym handling.
  * Explicitly configured `conduct` heteronym rule: noun -> `/K AA1 N D AH0 K T/` (e.g. E2a), verb -> `/K AH0 N D AH1 K T/` (e.g. E2b).
  * Neural fallback for OOV words (`andaroo`, `antidisestablishmentarianism`, split tokens `rhyth`, `m`).
  * Silence and punctuation handling mapping to `@sp` / `@spn`.
- Updated `English/text/symbols.py`:
  * Cleaned symbol export using relative imports, removed unused pinyin symbols.
- Updated `English/text/__init__.py`:
  * Seamless `text_to_sequence` supporting both `{ARPAbet}` and raw text via G2P.
- Unit tests: `tests/test_frontend_en.py` (5 tests covering all test sentences, conduct heteronym distinction, date/time expansion, OOV handling, and symbol validation — all passing).

## Inference & Variance Modeling (`English/inference.py`, `English/model/`)
- Created `English/inference.py`:
  * Config-driven self-contained inference script supporting `--checkpoint`, `--out_dir`, `--speaker`, `--device`, `--pitch_control`, `--energy_control`, `--duration_control`.
  * Loads pretrained HiFi-GAN vocoder from `vocoder/` directory dynamically.
  * Exports synthesized audio to 22050 Hz 16-bit PCM WAV.
  * Captures phoneme-level and frame-level variance arrays to `<out_dir>/variance_adaptor_plots/raw/<ID>.npz`.
  * Generates `<out_dir>/test_generated_audios/<ID>.json` with full metadata (normalized text, phonemes, symbol IDs, word-to-symbol mapping, RTF, symbols/frames count).
- Bug fixes in `English/model/modules.py`:
  * Line 124 & 149: Fixed reference repo bugs where `p_control` was erroneously passed into `get_energy_embedding` instead of `e_control`.
  * In `LengthRegulator.forward`: added default argument `max_len=None` and ensured dynamic `.to(x.device)` placement to prevent CPU/CUDA device mismatches.
  * In `LengthRegulator.expand`: added non-zero frame clamp so degenerate zero-duration predictions never collapse into 0-element tensors that crash decoder self-attention.
- Relative imports in `English/model/fastspeech2.py`:
  * Added fallback relative imports (`from transformer import ... except (ImportError, ValueError): from ..transformer ...`) so the model can be loaded from external scripts and package root without ModuleNotFoundError.

## Phase 5.5 Audit, Hardening & Deliverable Alignment
- Created `English/train.py`:
  * Self-contained training entrypoint supporting multi-speaker conditioning, gradient accumulation, Adam optimizer with warmup and multi-step annealing.
  * Full validation loop evaluating all validation samples (weighted mean loss).
  * Checkpoint persistence saving `model`, `optimizer`, `step`, `epoch`, and full RNG states (`python`, `numpy`, `torch`, `cuda`).
  * CSV loss logging (`logs/train_losses.csv`, `logs/val_losses.csv`) and TensorBoard event logging.
  * NaN / Inf loss detection dumping crash state to `checkpoints/crash_step_<step>.pth`.
  * Built-in `--smoke_test` mode running 100 fast steps on CPU with synthetic batch generation.
- Created `English/checkpoints/`:
  * Direct language-level checkpoint storage holding `best.pth` and `last.pth` per assignment specification.
- Updated `English/model/optimizer.py`:
  * Added `get_lr()`, `state_dict()`, and robust `load_state_dict()` preserving `current_step`.
- Updated `English/evaluate.py`:
  * Exact dynamic validation loss accumulation dividing by total evaluated samples rather than static dataset length.
- Seeding hardening:
  * Added standard Python `random.seed(1234)` alongside PyTorch and NumPy seed setters in all entrypoints.
