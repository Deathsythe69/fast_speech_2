# FastSpeech 2 Per-Sentence Variance Statistics — English

Generated from synthesized test sentences using predicted duration, pitch (F0), and energy.

| ID | # Sym | Dur (s) | Rate (sym/s) | Longest Sym (ms) | Mean Sil (ms) | F0 Mean±Std [Min, Max] (Hz) | F0 Range (st) | Declination (Hz) | Energy Mean±Std | Peak Energy Sym |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **E1** | 96 | 1.13 | 85.2 | `Z` (23.22 ms) | 11.61 ms | 219.81±23.03 [169.61, 281.8] | 8.79 st | -1.84 Hz | 39.1262±11.2721 | `M` (63.7761) |
| **E2a** | 61 | 0.71 | 86.1 | `DH` (11.61 ms) | 11.61 ms | 223.32±23.56 [161.29, 291.5] | 10.25 st | -7.53 Hz | 38.6336±12.8998 | `D` (79.3857) |
| **E2b** | 66 | 0.77 | 86.1 | `DH` (11.61 ms) | 11.61 ms | 222.14±20.91 [164.61, 267.4] | 8.4 st | 0.7 Hz | 41.5154±9.6864 | `K` (64.3332) |
| **E3** | 44 | 0.51 | 86.1 | `DH` (11.61 ms) | 11.61 ms | 223.42±25.06 [160.42, 281.31] | 9.72 st | -46.15 Hz | 38.4988±9.6937 | `EH1` (62.2724) |
| **E4** | 100 | 1.16 | 86.1 | `DH` (11.61 ms) | 11.61 ms | 218.22±25.86 [150.69, 285.15] | 11.04 st | -44.51 Hz | 39.5049±12.4837 | `ER0` (69.383) |

## Sentence Details
### E1
- **Raw Text**: The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings.
- **Normalized Text**: The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings.
- **Checkpoint**: `C:\Users\Panig\OneDrive\Documents\fast_speech_2\English\checkpoints\best.pth` (Step 1000) | **Speaker**: `speaker_0`
- **Speaking Rate**: 85.24 symbols/s | **Total Frames**: 97
- **F0 Declination**: -1.84 Hz (-0.15 st) relative to utterance mean (219.81 Hz)

### E2a
- **Raw Text**: The orchestra's conduct during the ceremony was praised by the audience.
- **Normalized Text**: The orchestra's conduct during the ceremony was praised by the audience.
- **Checkpoint**: `C:\Users\Panig\OneDrive\Documents\fast_speech_2\English\checkpoints\best.pth` (Step 1000) | **Speaker**: `speaker_0`
- **Speaking Rate**: 86.13 symbols/s | **Total Frames**: 61
- **F0 Declination**: -7.53 Hz (-0.59 st) relative to utterance mean (223.32 Hz)

### E2b
- **Raw Text**: The researchers will conduct an experiment to evaluate the proposed method.
- **Normalized Text**: The researchers will conduct an experiment to evaluate the proposed method.
- **Checkpoint**: `C:\Users\Panig\OneDrive\Documents\fast_speech_2\English\checkpoints\best.pth` (Step 1000) | **Speaker**: `speaker_0`
- **Speaking Rate**: 86.13 symbols/s | **Total Frames**: 66
- **F0 Declination**: 0.7 Hz (0.05 st) relative to utterance mean (222.14 Hz)

### E3
- **Raw Text**: The rhyth m of the orchestra changed unexpectedly.
- **Normalized Text**: The rhyth m of the orchestra changed unexpectedly.
- **Checkpoint**: `C:\Users\Panig\OneDrive\Documents\fast_speech_2\English\checkpoints\best.pth` (Step 1000) | **Speaker**: `speaker_0`
- **Speaking Rate**: 86.13 symbols/s | **Total Frames**: 44
- **F0 Declination**: -46.15 Hz (-4.01 st) relative to utterance mean (223.42 Hz)

### E4
- **Raw Text**: The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time.
- **Normalized Text**: The meeting will start at ten forty-five A M on twelve November twenty twenty-six, so andaroo should arrive on time.
- **Checkpoint**: `C:\Users\Panig\OneDrive\Documents\fast_speech_2\English\checkpoints\best.pth` (Step 1000) | **Speaker**: `speaker_0`
- **Speaking Rate**: 86.13 symbols/s | **Total Frames**: 100
- **F0 Declination**: -44.51 Hz (-3.95 st) relative to utterance mean (218.22 Hz)

#### Numeral & Time Word Breakdown
| Word | Phones | Span | Duration (ms) | Frames | Mean F0 (Hz) | Min/Max F0 (Hz) | Mean Energy |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `10:45` | `sp T EH1 N sp F AO1 R...` | [21:33] | 139.32 | 12 | 220.5 | [194.45, 285.15] | 35.4269 |
| `AM` | `sp T` | [21:23] | 23.22 | 2 | 201.35 | [201.35, 201.35] | 35.5899 |
| `12` | `T W EH1 L V` | [44:49] | 58.05 | 5 | 210.12 | [194.17, 239.45] | 37.4094 |
| `November` | `N OW0 V EH1 M B ER0` | [50:57] | 81.27 | 7 | 225.48 | [189.79, 268.81] | 43.9365 |
| `2026,` | `sp T W EH1 N T IY0 sp...` | [57:73] | 185.76 | 16 | 229.06 | [199.59, 256.06] | 36.9137 |

