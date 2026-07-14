# Sprint Retrospectives

## Sprint 1 retrospective

- **What went well:** Grouped re-split removed a 3,379-image leakage risk before any training time was spent.
- **What could improve:** Perceptual-hash grouping needed manual review of the largest clusters; start that review earlier.
- **Action item:** Adopt 'validate before train' as a standing rule for any dataset change.

## Sprint 2 retrospective

- **What went well:** Baseline and comparison models trained without NaN issues after AMP was disabled.
- **What could improve:** FP32 training on the 4GB GPU was slower than planned; epochs had to be trimmed.
- **Action item:** Record per-epoch timing in the experiment log to keep estimates realistic.

## Sprint 3 retrospective

- **What went well:** One shared inference engine kept image/video/webcam behaviour consistent.
- **What could improve:** streamlit-webrtc threading needed a lock around shared stats; found via testing.
- **Action item:** Write component tests before wiring UI callbacks.

## Sprint 4 retrospective

- **What went well:** Report and slides were generated from saved artefacts, so numbers match outputs exactly.
- **What could improve:** Packaging surfaced path issues late; the verification loop caught them.
- **Action item:** Run the packaging dry-run at the end of every sprint, not only at delivery.
