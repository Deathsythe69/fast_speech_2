# Project Spec — FastSpeech 2 TTS (English + Telugu)

Goal: Train two separate FastSpeech 2 TTS models (English on Svarah, Telugu on IndicTTS Telugu) based on ming024/FastSpeech2.
Deadline: 07-11-2026.

Deliverables checklist:
- train.py, preprocess.py, inference.py per language
- config.yaml per language
- checkpoints (.pt) per language
- training_loss.png and validation_loss.png per language
- test generated audios per language (English real; Telugu placeholders)
- variance_adaptor_plots/{duration,pitch,energy} per test sentence per language
- Technical_Report.pdf
- README.md
- requirements.txt
- test_sentences.json

Rules:
- ~5 h train + ~1 h val per language
- No train/val overlap
- Separate models and configs per language
- Document every change made to the reference repo
- Document Telugu-specific changes
- Preserve test sentence text exactly

Phases:
1. Setup
2. Dataset subsampling
3. Preprocessing/alignment
4. Training + loss plots
5. Inference + variance plots
6. Report + README

Decisions log:
- Split level: utterance-level groups; all rows sharing a normalized transcript are assigned to the same split (transcript-disjoint splits).
- Transcript-disjointness: no normalized transcript appears in both train and val; test sentences are excluded from both splits.
- Duration limits: utterances kept only if duration in [min_dur, max_dur] seconds (default 1.0–12.0 s); unreadable audio dropped.
- Target sample rate 22,050 Hz mono 16-bit PCM; Svarah is likely 16 kHz so audio is upsampled during conversion.
- Seed 1234 used to shuffle transcript groups for reproducible splits.
- Speaker selection policy: English val stratified per speaker proportional to total duration share; Telugu (single speaker) uses a plain global val-then-train split.

## Phase 3 Decisions Log:
- Self-Contained Language Directories: Cloned ming024/FastSpeech2 and distributed core architecture (`model/`, `text/`, `audio/`, `utils/`, `transformer/`, `preprocessor/`, `hifigan/` configs and code) into `English/` and `Telugu/` directories with dedicated `CHANGES.md` logs, removing unused dataset configs (AISHELL3, LibriTTS).
- English Text Frontend: Implemented full normalization for 4-digit years (`2026` -> twenty twenty-six), times (`10:45 AM` -> ten forty-five A M), dates (`12 November 2026` -> twelve November twenty twenty-six), and abbreviations while preserving apostrophes (`orchestra's`). Configured CMUdict + g2p_en with explicit POS-based heteronym override for `conduct` (noun `/K AA1 N D AH0 K T/` vs verb `/K AH0 N D AH1 K T/`), neural OOV fallback (`antidisestablishmentarianism`, `andaroo`, `rhyth`, `m`), and silence tokens (`@sp`, `@spn`).
- Telugu Text Frontend: Adopted a rule-based phonetic G2P representation over raw Unicode graphemes to unambiguously resolve inherent vowel `/a/` assignment, virama (`్`) suppression, and consonant conjunct decomposition (`త్త్` -> `t t a`, `ష్ఠ` -> `ss th_ a`). Handled Unicode NFC normalization, explicit ZWJ/ZWNJ detection/cleaning with event logging, and full Telugu numeral expansion (`2026` -> రెండు వేల ఇరవై ఆరు, `10:45` -> పది నలభై ఐదు, `12వ` -> పన్నెండవ). Documented homograph `పట్టు` (grip vs silk: identical segmental phonemes `/p a t_ t_ u/`, disambiguated by prosody and discourse role).
- Alignment Architecture: Implemented `tools/align.py` performing pre-alignment verification (22050 Hz mono audio, valid transcripts), automated MFA pronunciation lexicon generation (English ARPAbet dictionary, Telugu phonetic dictionary), Praat TextGrid generation, and alignment reporting (file counts, fail reasons, phone coverage, silence percentage, min/median/max phone durations, and >5% failure rate threshold enforcement).
- Feature Extraction and No-Leak Normalization: Standardized on 22050 Hz 80-channel mel-spectrograms (STFT filter 1024, hop 256, win 1024), PyWorld dio/stonemask pitch (F0), STFT L2 energy, and frame durations (`sum(durations) == T_mel`). In `preprocessor.py`, eliminated reference repo's validation data leakage by computing scalers and `stats.json` min/max/mean/std strictly on the TRAIN split only.
- Configuration Divergences: English uses multi-speaker mode (`multi_speaker: True`), broader F0 range [50, 500] Hz for male/female speakers, and 78-symbol ARPAbet inventory. Telugu uses single-speaker mode (`multi_speaker: False`), female F0 range [100, 500] Hz, and 58-symbol Telugu phonetic inventory. Both utilize max sequence length of 1000 frames.

## Phase 5 Decisions Log:
- Speaker Choice: For English, multi-speaker mode is enabled; inference defaults to the dominant speaker with the most training duration read from `data/english/train.csv` (fallback `speaker_0`). For Telugu, single-speaker mode (`multi_speaker: False`) uses the fixed female speaker (`default` / speaker ID 0).
- Denormalization Method: In `inference.py` and variance plotting, log-scale pitch is denormalized via $F_0 = \exp(p_{pred} \cdot \sigma_p + \mu_p)$ where $[\mu_p, \sigma_p]$ are loaded from `preprocessed_data/<lang>/stats.json` computed strictly on train data. Exponents are safely bounded to $[\ln(20), \ln(1000)]$ to prevent numerical overflow on extreme outliers. Energy is linearly denormalized via $e = e_{pred} \cdot \sigma_e + \mu_e$.
- Plot Design & Layout: Shared two-panel layout at 200 DPI: top panel displays phoneme-level predictions (duration in ms with frame count annotations, pitch in Hz with unvoiced/silence tokens clearly marked, energy in amplitude bars) with word boundaries and word labels overhead. Bottom panel overlays the frame-level predicted mel spectrogram with continuous pitch ($F_0$) or energy curves and vertical phone boundary lines. Sentences with $>40$ symbols are wrapped into segmented rows for legible font display.
- Font Solution: Registered Windows system font `Nirmala.ttc` (`NirmalaUI`) and fallback `assets/fonts/NotoSansTelugu-Regular.ttf` with Matplotlib's font manager. This resolves both Latin and Telugu script glyphs simultaneously without tofu/box artifacts or missing glyph warnings.
- Comparative Phonetic Analysis: Implemented specialized comparative plotting: `conduct_E2a_vs_E2b.png` isolates noun vs verb stress shift across duration, pitch, and energy for `/K AA1 N D AH0 K T/` vs `/K AH0 N D AH1 K T/`. `pattu_T2a_vs_T2b.png` compares the homograph root `"పట్టు"` (`p a t_ t_ u`) between physical grip (T2a) and silk saree compound (T2b).
- Validation Protocol: Quantitative validation (`tools/eval_val.py`) computes teacher-forced metrics (duration MAE in frames and ms, voiced pitch RMSE and Pearson $r$, energy RMSE and Pearson $r$, mel L1 postnet and pre-postnet) and free-running total duration error (MAPE % and MAE ms). On active GPU training systems, evaluation executes on CPU across a sampled subset (up to 200 utterances) to avoid GPU OOM or slowing down training processes.
- Reference Repo Bugfixes & Deviations: Corrected the reference repo's bug in `model/modules.py` line 124 where `p_control` was erroneously passed to `get_energy_embedding` instead of `e_control`. Enhanced `LengthRegulator.forward` with `max_len=None` default and `.to(x.device)` dynamic tensor placement, and guarded `expand` with zero-frame clamping to ensure self-attention never crashes on degenerate all-zero duration predictions. Fixed relative imports in `fastspeech2.py` for standalone execution.
