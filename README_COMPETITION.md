# Competition submission: multimodal sentiment and explanation

This branch is isolated from `robust-msa-v2`. It keeps the existing RobustMSA
research code, but uses a smaller, validated model as the default submission
path. Do not report an unvalidated deep model as the winning result.

## Task and data contract

- Attachment 2 supplies the only supervised data: 3,395 training, 728
  validation, and 727 held-out test clips. Its `unaligned_50.pkl` representation
  is used consistently in training and inference. Do not mix it with aligned
  features.
- Polarity is `Negative` when the ground-truth intensity is less than zero,
  `Neutral` at exactly zero, and `Positive` when greater than zero.
- MAE and Pearson correlation evaluate intensity; accuracy and macro-F1
  evaluate polarity. There are 30 unlabeled Attachment 3 clips and 20
  unlabeled Attachment 4 clips. They are never used for tuning.
- Attachment 3 has `raw_text`, audio, and vision features, but no BERT text
  embeddings. Attachment 4 has embeddings in its PKLs. The submission entry
  point reconstructs only missing text embeddings and checks the local BERT
  checkpoint against Attachment 2 before prediction.

## Why the default is a regularized baseline

The previous complex RECAP/RobustMSA run had a large validation-to-test gap.
On only 3,395 training examples, a high-capacity model can fit the training
split while generalizing poorly. `linear_baseline.py` pools observed (nonzero)
text/audio/vision frames, standardizes features using **training data only**,
and fits a ridge model for continuous intensity and three ridge class scores.
Feature groups and regularization are chosen on validation; a small polarity
decision grid is also fitted on validation. The held-out test set is reported
only after selection. No test or unlabeled labels enter fitting.

Measured on the local supplied `unaligned_50.pkl`:

| Split | MAE ↓ | Pearson Corr ↑ | Polarity accuracy ↑ | Macro-F1 ↑ |
|---|---:|---:|---:|---:|
| Validation | 0.5896 | 0.6137 | 0.6168 | 0.6097 |
| Held-out test | 0.6562 | 0.6458 | 0.6547 | 0.6278 |

These are reproducible *baseline* measurements, not a guaranteed score on
the unlabeled competition clips. The old screenshot reported test MAE 0.8921
and Corr -0.0862, but that run's exact protocol may differ, so it is not a
controlled ablation. Run `linear_baseline.py train` to regenerate the model.
The reported continuous scores are projected to match the task definition:
neutral is exactly zero, positive is above zero, and negative is below zero.
The unprojected regression value is retained in explanation files.

## Reproduce and submit

Run from the repository root. The scripts need Python with NumPy; materializing
text also needs PyTorch, Transformers, and the local `bert-base-uncased` model
directory. Use the checkpoint supplied with this project or an exactly matching
public one. A mismatched checkpoint causes a hard failure.
`artifacts/linear_baseline.npz` is the already-fitted 46 KB model parameter
file, so retraining is optional. The BERT weights are external and must be
provided separately; the script checks their numerical compatibility.

```powershell
python linear_baseline.py train --data "PATH_TO_ATTACHMENT2/unaligned_50.pkl" --output "outputs/linear_baseline.npz"
python submit_competition.py --model "artifacts/linear_baseline.npz" --bert_model "PATH_TO_BERT/bert-base-uncased" --reference_pkl "PATH_TO_ATTACHMENT2/unaligned_50.pkl" --attachment3_dir "PATH_TO_ATTACHMENT3/unaligned" --attachment4_dir "PATH_TO_ATTACHMENT4/unaligned" --output_dir "outputs/final"
```

The second command validates counts, unique IDs, polarity names, and the
[-3, 3] intensity range. It produces separate compact predictions and
detailed explanation CSVs for both attachments. Keep the checkpoint and
the final CSVs together when handing in the source, results, and parameters.
The tested results are already copied into `submission/`; rerunning the
command replaces only files under the output directory you choose.

## Interpretation

The linear score is decomposed exactly into standardized text/audio/vision
contributions relative to the training mean. The polarity margin is the
selected class score minus the runner-up score when class scores select the
label. `main_modality` is the largest absolute contribution under that rule;
it is model reliance, **not** a causal claim. A key window's signed `delta`
is how much the relevant score falls or rises when that window is removed
from its modality's mean. Text locations are BERT token positions; audio and
video locations are feature-frame windows with approximate relative
positions. When Attachment 4 MP4s are available, approximate seconds are
computed from their durations; feature frames are not guaranteed to align
exactly with those seconds. These are not guaranteed human reasons.
Text token IDs are decoded using the same BERT vocabulary.

`predict_robust.py` also offers actual modality-removal deltas for an existing
RobustMSA checkpoint. Attention weights alone are not treated as evidence of
causality. The regularized baseline is the default until a stronger model
passes the same held-out evaluation.

## Research context and limits

[P-RMF (ACL 2025)](https://aclanthology.org/2025.acl-long.1075/) addresses
missing modalities with learned latent proxies and uncertainty-aware fusion.
That design is relevant to this problem, but its reported benchmark numbers
are not directly comparable to these supplied splits. Here, observed-row
pooling is deliberately simpler and can be audited. It can miss temporal
interactions, while generated BERT features are a dependency for Attachment 3.
Before submitting, review the explanation CSVs against the Attachment 4
videos and document the pretrained model version and any manual interpretation.
