# RECAP: REcovering Coherent Affective Patterns Addressing Modality Missing in Multimodal Sentiment Analysis

### 🎉 Congrats! RECAP has been accepted by AAAI 2026 as an Oral Presentation

Pytorch implementation of the paper:
> **[Recovering Coherent Affective Patterns Addressing Modality Missing in Multimodal Sentiment Analysis](https://ojs.aaai.org/index.php/AAAI/article/view/39349/43310)**

> This repository is a reorganized version of our codebase for public release. If you find any bugs or issues, please feel free to contact me.

## Content

- [Overview](#overview)
- [Data Preparation](#data-preparation)
- [Environment](#environment)
- [Training & Evaluation](#training--evaluation)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)

## Overview

RECAP is a two-stage framework for multimodal sentiment analysis under modality missing settings:

1. Stage 1: modality completion
2. Stage 2: fusion prediction

![RECAP Framework](assets/RECAP_framework.png)


## Data Preparation

Please prepare the datasets in pickle format and place them under the `data/` directory in this repository:

- `data/mosi/unaligned_50.pkl`
- `data/mosei/unaligned_50.pkl`
- `data/sims/unaligned_39.pkl`

For downloading and processing MOSI, MOSEI, and SIMS, please see [MMSA](https://github.com/thuiar/MMSA).

If you store them elsewhere, update the paths in `configs/*.yaml`.

## Environment

The basic training environment for the results in the paper is Python 3.9.18, PyTorch 2.2.2, and CUDA 12.2 on NVIDIA RTX 4090 GPUs with 24GB memory.


## Training & Evaluation

### Step 1: Train the stage 1 completion model

Run stage 1 first to train the modality completion module:

```bash
python train.py --stage completion --config_file configs/train_mosi.yaml
```

After training finishes, a stage 1 checkpoint will be saved under `ckpt/<dataset_name>/` folder.


### Step 2: Train the stage 2 fusion model and evaluate

Then use the saved stage 1 checkpoint to train the fusion prediction stage:

```bash
python train.py \
  --stage fusion_prediction \
  --config_file configs/train_mosi.yaml \
  --stage1_ckpt ckpt/mosi/<your_stage1_checkpoint>.pth
```

If you prefer, you can also pass the stage 1 checkpoint suffix through `--time`, which will be expanded internally to `stage1_modules_<time>.pth`.

You can also obtain the validation and test results directly from the stage 2 training logs, and set `--missing_rate_eval_test` to evaluate under different missing rates.

For the joint polarity-classification and intensity-regression task, stage 2
automatically computes inverse-frequency class weights from the training split.
It uses `bert_lr` for BERT and `lr` for the remaining modules, and saves three
validation-selected checkpoints:

- `best_valid_polarity_f1_seed<seed>.pth` (classification)
- `best_valid_mae_seed<seed>.pth` (regression)
- `best_valid_joint_seed<seed>.pth` (joint score, used for the final test report)

The joint validation score is
`Macro-F1 + selection_corr_weight * Corr - selection_mae_weight * MAE`.
The test split is evaluated once after training with the selected joint
checkpoint. The relevant optional `base` configuration keys are `bert_lr`,
`task_modal`, `head_dropout`, `early_stopping_patience`,
`selection_corr_weight`, and `selection_mae_weight`.
Set `checkpoint_tag` to keep experiments separate; the continuous adaptation
uses `continuous_v3` by default so earlier checkpoints are not overwritten.

### Continuous-intensity competition adaptation

The regression branch keeps each pooled modality instead of reducing the three
modalities to one weighted average. It combines residual multimodal fusion with
a bounded `[-3, 3]` prediction and optimizes Smooth-L1/MAE, concordance
correlation, pairwise order, and polarity-consistency objectives. This directly
aligns training with the competition MAE and Pearson-correlation metrics.

Stage 1 checkpoints now contain BERT, all three feature projectors, and the
completion generator. Older checkpoints contain only the generator and are
accepted with a warning, but retraining Stage 1 is strongly recommended because
the old checkpoint connects a trained generator to newly initialized feature
projectors during Stage 2.

### Small-data regularized Stage 2

For small competition datasets, set `freeze_stage1_backbone: true` to keep the
Stage 1 BERT/projectors/generator fixed and train only the fusion and prediction
modules. `recovery_mode: residual` preserves observed features while adding a
scaled completion residual. Stage 2 also supports a separate learning rate and
weight decay, modality/fusion/head dropout, label smoothing, and shorter early
stopping. These settings reduce the train-validation gap without retraining
Stage 1. Use `--eval_checkpoint <path>` to evaluate a saved checkpoint without
training it again.

### RobustMSA: missing-aware, interpretable competition model

The two-stage RECAP pipeline above is kept unchanged.  `RobustMSA` is a
separate, lighter single-stage model for the E-problem (CMU-MOSEI subset,
Attachment 2 `unaligned_50.pkl`) that targets the train/validation gap:

- **Same missing definition everywhere.** A position is *missing* when its
  feature row is all zero (the Attachment 3 definition).  Missing positions get
  a learned embedding and are excluded from evidence pooling.
- **Attachment-3-style augmentation.** Training zeros random *contiguous*
  spans (10-60 % of the valid length, up to 3 spans) and occasionally a whole
  modality; the old pipeline instead dropped 50 % of random tokens.
- **Train-set feature standardization** over observed rows (COVAREP/Facet
  scales differ by orders of magnitude); `nan`/`inf` are zeroed.
- **Small model, no unsupervised stage.** Per-modality 2-layer Transformers
  (hidden 128, audio/vision pooled from 500 to 100 steps), a reliability-aware
  modality gate, and regression + polarity heads with unimodal auxiliary heads.
- **Validation matches both special tests.** Every epoch is scored on the
  complete valid split (Attachment 4) and on a fixed seeded contiguous-missing
  copy (Attachment 3). The classification-first selection objective is the
  view average of `0.6 * Macro-F1 + 0.4 * Accuracy`.
- **Polarity decision on validation.** After training, class log-biases or
  intensity thresholds are chosen by the same validation-only objective and
  stored in the checkpoint. The test split is evaluated once, after the final
  loss-weight configuration has been selected.

```bash
# train a few seeds (config: configs/robust_mosei.yaml)
for s in 1111 2222 3333; do python train_robust.py --seed $s; done

# ensemble metrics, ensemble polarity rule, missing-pattern analysis
# (modality set x begin/middle/end/random x 10-100 % duration) for Problem 2
python analyze_robust.py --checkpoints "ckpt/robust_mosei/robust_robust_v1_seed*.pth"

# Attachment 3 / 4 predictions with explanations
python predict_robust.py --checkpoints "ckpt/robust_mosei/robust_robust_v1_seed*.pth" \
  --decision_json outputs/ensemble_decision.json \
  --input_dir <attachment3_dir> --output_csv outputs/attachment3_predictions.csv
```

### Loss-weight tuning and sensitivity

`run_loss_tuning.py` compares the baseline with the recommended A/B/C joint
settings and reproduces the one-factor sensitivity curves for classification,
unimodal auxiliary and Pearson loss weights. Tuning uses only the Attachment 2
validation split. A candidate remains eligible when its mean MAE is at most
0.01 above baseline and its mean Pearson correlation is at most 0.01 below
baseline.

```bash
# 16 unique configurations x 3 seeds; finished runs are skipped on restart
python run_loss_tuning.py --gpus 0 1 --jobs_per_gpu 1

# list commands without starting training
python run_loss_tuning.py --dry_run

# rebuild summaries and the sensitivity figure from completed runs
python run_loss_tuning.py --summarize_only
```

The script writes `loss_combination_summary.csv`, `loss_recommendation.json`,
`loss_sensitivity_summary.csv` and `loss_weight_sensitivity_valid.png` under
`outputs/loss_tuning/`.

The prediction CSV contains polarity, intensity, class probabilities, the main
modality, per-modality contribution weights, unimodal intensities, observed
ratios and the top-k key evidence positions per modality (text word pieces;
audio/vision frame ranges with their relative position in the clip, which maps
to the video time as `relative position x clip duration`).

`text_source: precomputed` uses the 50x768 `text` field, so text missing spans
in Attachment 3 are detected exactly like audio/vision.  `text_source: bert`
fine-tunes the top 4 BERT layers from `text_bert` instead; it needs
`text_bert` in the special-test files as well.



## Acknowledgement

This repository is built upon the official codebase of [LNLN](https://github.com/Haoyu-ha/LNLN). We sincerely thank the authors for their open-source contribution.


## Citation

If you find this repository useful for your research, please cite our paper:

```bibtex
@inproceedings{huang2026recovering,
  title={Recovering coherent affective patterns: Addressing modality missing in multimodal sentiment analysis},
  author={Huang, Huiting and Gong, Tieliang and He, Kai and Wen, Wen and Zhang, Weizhan and Feng, Mengling},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={26},
  pages={21957--21965},
  year={2026}
}
```
