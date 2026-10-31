# Quantitative Validation Metrics — English *(Provisional dry-run on validation subset; CPU mode)*

- **Checkpoint**: `C:\Users\Panig\OneDrive\Documents\fast_speech_2\checkpoints\english\last.pth` (Step 1000)
- **Evaluation Device**: `CPU`
- **Evaluated Utterances**: 25
- **Status**: `Provisional`

## Summary Table

| Evaluation Mode | Metric | Value | Unit | Target / Best |
|:---|:---|:---:|:---:|:---:|
| **Teacher-Forced** | Duration MAE | 1.989 (23.09 ms) | frames (ms) | Lower |
| **Teacher-Forced** | Voiced Pitch RMSE | 624.28 | Hz | Lower |
| **Teacher-Forced** | Voiced Pitch Pearson $r$ | 0.0 | correlation | Higher (~1.0) |
| **Teacher-Forced** | Energy RMSE | 13.5455 | amplitude | Lower |
| **Teacher-Forced** | Energy Pearson $r$ | 0.0073 | correlation | Higher (~1.0) |
| **Teacher-Forced** | Mel Spectrogram L1 (PostNet) | 5.0815 | L1 norm | Lower |
| **Teacher-Forced** | Mel Spectrogram L1 (Pre-PostNet) | 5.0861 | L1 norm | Lower |
| **Free-Running** | Total Duration Error (MAPE) | 79.43% | % | Lower |
| **Free-Running** | Total Duration Error (MAE) | 1399.23 | ms | Lower |

> [!NOTE]
> When running with active GPU training, evaluation executes on CPU across a sampled subset (up to 200 utterances) to ensure training processes are not impeded.
