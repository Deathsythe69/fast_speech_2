# Quantitative Validation Metrics — Telugu *(Provisional dry-run on validation subset; CPU mode)*

- **Checkpoint**: `C:\Users\Panig\OneDrive\Documents\fast_speech_2\checkpoints\telugu\last.pth` (Step 1000)
- **Evaluation Device**: `CPU`
- **Evaluated Utterances**: 25
- **Status**: `Provisional`

## Summary Table

| Evaluation Mode | Metric | Value | Unit | Target / Best |
|:---|:---|:---:|:---:|:---:|
| **Teacher-Forced** | Duration MAE | 1.58 (18.34 ms) | frames (ms) | Lower |
| **Teacher-Forced** | Voiced Pitch RMSE | 601.26 | Hz | Lower |
| **Teacher-Forced** | Voiced Pitch Pearson $r$ | 0.0 | correlation | Higher (~1.0) |
| **Teacher-Forced** | Energy RMSE | 15.9428 | amplitude | Lower |
| **Teacher-Forced** | Energy Pearson $r$ | -0.0121 | correlation | Higher (~1.0) |
| **Teacher-Forced** | Mel Spectrogram L1 (PostNet) | 5.0241 | L1 norm | Lower |
| **Teacher-Forced** | Mel Spectrogram L1 (Pre-PostNet) | 5.0226 | L1 norm | Lower |
| **Free-Running** | Total Duration Error (MAPE) | 68.05% | % | Lower |
| **Free-Running** | Total Duration Error (MAE) | 1035.61 | ms | Lower |

> [!NOTE]
> When running with active GPU training, evaluation executes on CPU across a sampled subset (up to 200 utterances) to ensure training processes are not impeded.
