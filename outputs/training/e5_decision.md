# Final model (E5) - configuration decision

The final model is **not** a fresh run with hand-picked settings. It continues
fine-tuning the strongest existing checkpoint (`outputs/training/e1_baseline_v8n/weights/best.pt`) using the
single change that measurably beat the control in the probe study below.

## Probe study (equal budget, one variable each, same seed and data)

Control (`e4d_probe_baseline`): mAP@0.5 = 0.3266, mAP@0.5:0.95 = 0.1367, recall = 0.3541 at 5 epochs.

| Probe | Change | mAP@0.5 | mAP@0.5:0.95 | Recall | Δ mAP@0.5:0.95 | Verdict |
|---|---|---|---|---|---|---|
| `e4a_probe_adamw` | AdamW optimizer with lr0=1e-3 | 0.3732 | 0.1636 | 0.3754 | +0.0269 | **adopt** |
| `e4b_probe_augment` | stronger photometric + scale augmentation | 0.3026 | 0.1264 | 0.3552 | -0.0102 | **reject** |
| `e4c_probe_loss` | classification-loss weight 0.5 -> 1.0 | 0.3968 | 0.1695 | 0.3850 | +0.0328 | **adopt** |

## Decision

- **Adopted:** classification-loss weight 0.5 -> 1.0 (`e4c_probe_loss`), the largest measured improvement (+0.0328 mAP@0.5:0.95 against the control).
- **Rejected:** stronger photometric + scale augmentation - it made things worse (-0.0102). Reported as measured.
- **Not combined:** AdamW optimizer with lr0=1e-3 (`e4a_probe_adamw`) also helped on its own, but the probes tested it in isolation. Stacking two changes whose interaction was never measured would make the final result unattributable, so it was left out.
- **Continuation LR:** lr0 = 0.002. The run starts from a trained checkpoint, so the default peak LR (0.01) would partly undo it. This is standard fine-tuning practice, not a tuned value, and it is stated rather than hidden.

## Command actually run

```bash
python scripts/train_final.py --final \
    model=outputs/training/e1_baseline_v8n/weights/best.pt \
    cls=1.0 \
    lr0=0.002 \
```

## Caveat we state out loud

Probes were run at a short budget because the GPU is a 4GB GTX 1650 Ti on which
AMP is auto-disabled (FP32, ~2x slower). Short probes rank configurations *under
a short schedule*; the learning-rate schedule depends on total epochs, so a
setting that wins at 5 epochs is not guaranteed to win at 50. With more compute,
the honest design repeats the probes at full length.
