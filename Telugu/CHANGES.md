# Changes Log — Telugu Model

Tracking all modifications from ming024/FastSpeech2 reference repository.

## Setup
- Copied core architecture: `model/`, `text/`, `audio/`, `utils/`, `transformer/`, `preprocessor/`, `hifigan/` (configs and code only, excluding large weight archives).
- Copied entry scripts: `dataset.py`, `preprocess.py`, `evaluate.py`, `synthesize.py`.
- Removed unused dataset-specific files: `preprocessor/aishell3.py`, `preprocessor/libritts.py`, `config/AISHELL3`, `config/LibriTTS`.

## Text Frontend (`Telugu/text/`)
### Grapheme vs Rule-Based G2P Decision
- Decision: Implemented a **rule-based Telugu G2P** phonetic representation.
- Justification:
  * Telugu orthography is strictly phonetic and abugida-based.
  * Inherent vowel `/a/` handling: A grapheme-only model requires the network to implicitly learn when a consonant lacks a vowel sign vs when virama (`్`) suppresses it. A rule-based G2P explicitly handles virama suppression and vowel matra substitution.
  * Conjuncts (e.g. `త్త్`, `ర్ధ`, `ష్ఠ`): Decompose cleanly into sequential phonetic consonant segments (`t t a`, `r dh a`, `sh th_ a`), avoiding Unicode rendering and multi-codepoint ordering inconsistencies.
  * Direct alignment compatibility: Aligns directly with Montreal Forced Aligner (MFA) phonetic dictionaries and FastSpeech 2 phone duration targets.

### Homograph / Ambiguity Documentation
- **పట్టు (pattu)**:
  * In Telugu, "పట్టు" is a homograph:
    - Noun sense 1: "grip" / "tenacity" (e.g. T2a: "తాడుపై అతని పట్టు చాలా బలంగా ఉంది." -> his grip on the rope was very strong).
    - Noun sense 2: "silk" (e.g. T2b: "ఆమె పట్టుచీరను ధరించి వేడుకకు వచ్చింది." -> she wore a silk saree to the ceremony).
    - Verb root: "to catch" / "to hold".
  * Segmental phonetics: Orthographically and phonetically, all senses share identical segmental phone representations `/p a t_ t_ u/`. Prosodic pitch and duration variations in natural speech stem from clausal prominence and discourse context rather than distinct segmental phonemes.

### Modules Implemented
- `Telugu/text/normalizer.py`:
  * Unicode NFC normalization.
  * Explicit ZWJ (`\u200d`) and ZWNJ (`\u200c`) handling with event logging. Replaces `్\u200c` with standard virama `్`.
  * Numeral, time, and ordinal expansion:
    - `2026` -> "రెండు వేల ఇరవై ఆరు"
    - `10:45` -> "పది నలభై ఐదు"
    - `12వ` -> "పన్నెండవ"
- `Telugu/text/symbols.py`:
  * Fixed phonetic inventory of Telugu vowels, consonants, modifiers (`@m`, `@h`), and silence tokens (`@sp`, `@spn`, `@sil`).
- `Telugu/text/g2p.py`:
  * Deterministic rule-based phonetic parser.
- `Telugu/text/__init__.py`:
  * Exposes `text_to_sequence` (supports `{...}` phone syntax and raw Telugu text).
- Unit tests: `tests/test_frontend_te.py` (4 tests covering numeral expansion, ZWJ/ZWNJ handling, homograph 'పట్టు', and no-unknown-symbols across all 5 test sentences — all passing).
## Inference & Variance Modeling (Telugu/inference.py, Telugu/model/)
- Created Telugu/inference.py:
  * Config-driven self-contained inference script supporting --checkpoint, --out_dir, --device, --pitch_control, --energy_control, --duration_control.
  * Pretrained HiFi-GAN vocoder integration from ocoder/ directory.
  * Exports synthesized audio to 22050 Hz 16-bit PCM WAV.
  * Captures phoneme-level and frame-level variance arrays to <out_dir>/variance_adaptor_plots/raw/<ID>.npz.
  * Generates <out_dir>/test_generated_audios/<ID>.json with full metadata (normalized text, phonemes, symbol IDs, word-to-symbol mapping, RTF, symbols/frames count).
- Bug fixes in Telugu/model/modules.py:
  * Line 124: Fixed reference repo bug where p_control was erroneously passed into get_energy_embedding instead of e_control.
  * In LengthRegulator.forward: added default argument max_len=None and ensured dynamic .to(x.device) placement to prevent CPU/CUDA device mismatches.
- Relative imports in Telugu/model/fastspeech2.py:
  * Added fallback relative imports (rom transformer import ... except ImportError: from ..transformer ...) so the model can be loaded from external scripts and package root without ModuleNotFoundError.
