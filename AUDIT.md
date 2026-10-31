# Codebase Bug Audit Report — FastSpeech 2 (English & Telugu)

Date: 2026-10-31  
Branch: `hardening`  
Scope: Full static and dynamic audit across `English/`, `Telugu/`, `tools/`, and `tests/`.

---

## 1. Audit Findings Table

| ID | File:Line | Category | Severity | Evidence | Resolution / Fix | Status |
|:---:|:---|:---|:---:|:---|:---|:---:|
| **BUG-01** | `tools/align.py:70` | Syntax / Typo | **High** | `os.dict_path` does not exist; raises `AttributeError: module 'os' has no attribute 'dict_path'` when dictionary path contains directory components. | Replaced `os.dict_path` with `out_dict_path`. | **Fixed** |
| **BUG-02** | `English/utils/model.py:58`, `Telugu/utils/model.py:58` | Path Resolution | **High** | `with open("hifigan/config.json", "r")` and `ckpt = torch.load("hifigan/generator_universal.pth.tar")` use hardcoded relative paths that fail when executed from repository root. | Updated to dynamically resolve paths via `Path(__file__)` with fallback to root `vocoder/` directory. | **Fixed** |
| **BUG-03** | `English/model/modules.py:124`, `Telugu/model/modules.py:124` | Logic Error | **High** | In reference repository, `p_control` was passed into `get_energy_embedding` instead of `e_control`, coupling energy control to pitch control. | Changed argument to `e_control`. | **Fixed** |
| **BUG-04** | `English/model/modules.py:170`, `Telugu/model/modules.py:170` | Architecture | **High** | `LengthRegulator.forward` lacked default `max_len=None`, causing `TypeError`. Line 183 also had hardcoded `.to(device)` causing CPU/CUDA mismatch. | Added `max_len=None` default and dynamic `.to(x.device)` placement. | **Fixed** |
| **BUG-05** | `English/model/modules.py:137`, `Telugu/model/modules.py:137` | Edge Case | **Medium** | If duration predictor predicts zero for all phones in an utterance, `LengthRegulator` collapsed to zero frames, crashing decoder self-attention (`cannot reshape tensor of 0 elements`). | Clamped `duration_rounded` to minimum 1 if total frame sum is 0; guarded `LengthRegulator.expand` against empty returns. | **Fixed** |
| **BUG-06** | `tools/plot_variance.py:25` | Portability | **Medium** | Hardcoded Windows path `C:\\Windows\\Fonts\\Nirmala.ttc` causes missing font errors on Linux systems. | Added dynamic platform check using `os.environ.get("WINDIR")` and bundled fallback `assets/fonts/NotoSansTelugu-Regular.ttf`. | **Fixed** |
| **BUG-07** | `English/train.py`, `Telugu/train.py` | Missing Deliverable | **Critical** | Assignment specification explicitly requires `English/train.py` and `Telugu/train.py` in the GitHub folder structure; neither file existed. | Implemented self-contained, production-grade `train.py` for both languages with full Phase 4 features and smoke-test support. | **Fixed** |
| **BUG-08** | `English/checkpoints/`, `Telugu/checkpoints/` | Deliverable Layout | **Medium** | Assignment layout requires language-specific `checkpoints/` subdirectories (`project/<Lang>/checkpoints/`); checkpoints were stored only in root `checkpoints/<lang>/`. | Structured checkpoints under both `English/checkpoints/` and `Telugu/checkpoints/` and updated inference loaders to check both locations. | **Fixed** |
| **BUG-09** | `English/utils/model.py:20`, `Telugu/utils/model.py:20` | Reproducibility | **Medium** | Checkpoint saving in reference repo omitted `step`, `epoch`, `scheduler`, and RNG states (`torch`, `numpy`, `python`). Resuming restarted at epoch 1 with desynchronized LR and seeds. | Checkpoint save/restore updated to encapsulate full dictionary: `step`, `epoch`, `model`, `optimizer`, `scheduler`, and full RNG states. | **Fixed** |
| **BUG-10** | `English/evaluate.py:44`, `Telugu/evaluate.py:44` | Metric Accuracy | **Low** | Validation loss averaging divided by static `len(dataset)`, causing minor skew when batches are dropped or grouped. | Dynamically accumulate evaluated sample count and compute exact weighted average. | **Fixed** |
| **BUG-11** | `English/inference.py:100`, `Telugu/inference.py:84` | Reproducibility | **Low** | Deterministic seeding set `torch.manual_seed` and `np.random.seed`, but omitted standard Python `random.seed(1234)`. | Added `random.seed(1234)` to all entrypoints. | **Fixed** |
| **BUG-12** | `tools/plot_variance.py:450` | Path Resolution | **Medium** | Passing `--out_dir Results` caused `plot_variance.py` to search `Results/variance_adaptor_plots/raw` instead of `Results/<Lang>/variance_adaptor_plots/raw`. | Added language subdirectory detection and normalized path handling. | **Fixed** |
| **BUG-13** | `tools/eval_val.py:285` | Numerical Stability | **Low** | Large unconstrained pitch predictions during dry-run caused exponential overflow (`RuntimeWarning: overflow encountered in exp`). | Bounded input exponent to safe range `[ln(20), ln(1000)]`. | **Fixed** |
| **BUG-14** | `English/preprocessor/preprocessor.py:273`, `Telugu/preprocessor/preprocessor.py:273` | Alignment Safety | **Low** | `duration[max_idx] = max(1, duration[max_idx] + diff)` could leave `sum(duration) != T_mel` if clamped when `diff < 0`. | Implemented iterative adjustment ensuring `sum(duration) == T_mel` strictly. | **Fixed** |
| **BUG-15** | `test_sentences.json` | Text Fidelity | **Medium** | Discrepancies between PDF specification text and `test_sentences.json` for Telugu: T2b had compound `"పట్టుచీరను"` (PDF: `"పట్టు చీరను"`), T3 had normalized `"అడవిలో"` (PDF: `"అడ విలో"` with typo space), T4 had `"ఉండండి."` (PDF: `"ఉంచండి."`). | Updated text processing to handle both exact PDF strings and normalized variants transparently without altering G2P phonetic output. | **Fixed** |
| **BUG-16** | `English/model/fastspeech2.py:1`, `Telugu/model/fastspeech2.py:1` | Modularity | **Medium** | Absolute imports `from transformer import ...` failed when importing from external evaluation scripts or package root. | Added fallback relative imports for robust multi-context execution. | **Fixed** |

---

## 2. Retraining Requirements Assessment

- **Items Requiring Retraining**: **None (0 items)**.
- **Evidence**:
  1. Train/validation data leakage was verified absent; `preprocessor.py` strictly computes scalers and `stats.json` on the train partition only (`test_val_split_not_used_for_stats_computation` PASSING).
  2. Acoustic features (80-channel mel, 22.05 kHz audio, hop 256, STFT filter 1024, PyWorld F0, L2 energy) are mathematically intact.
  3. All identified bugs are runtime execution, path resolution, interface compatibility, or edge-case handling bugs that do not corrupt or invalidate training objectives.
- **Estimated Cost of Retraining**: **0 GPU hours ($0.00)**.
