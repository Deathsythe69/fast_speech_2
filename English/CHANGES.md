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
## Inference & Variance Modeling (English/inference.py, English/model/)
- Created English/inference.py:
  * Config-driven self-contained inference script supporting --checkpoint, --out_dir, --speaker, --device, --pitch_control, --energy_control, --duration_control.
  * Loads pretrained HiFi-GAN vocoder from ocoder/ directory.
  * Exports synthesized audio to 22050 Hz 16-bit PCM WAV.
  * Captures phoneme-level and frame-level variance arrays to <out_dir>/variance_adaptor_plots/raw/<ID>.npz.
  * Generates <out_dir>/test_generated_audios/<ID>.json with full metadata (normalized text, phonemes, symbol IDs, word-to-symbol mapping, RTF, symbols/frames count).
- Bug fixes in English/model/modules.py:
  * Line 124: Fixed reference repo bug where p_control was erroneously passed into get_energy_embedding instead of e_control.
  * In LengthRegulator.forward: added default argument max_len=None and ensured dynamic .to(x.device) placement to prevent CPU/CUDA device mismatches.
- Relative imports in English/model/fastspeech2.py:
  * Added fallback relative imports (rom transformer import ... except ImportError: from ..transformer ...) so the model can be loaded from external scripts and package root without ModuleNotFoundError.
