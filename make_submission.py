"""Build the Problem 2 / Problem 3 submission attachment (zip, <= 50 MB).

    python make_submission.py --bert_path <bert-base-uncased dir> \
        --att3_dir <Attachment 3>/未对齐版本 --att4_dir <Attachment 4>/未对齐版本

Steps: copy only the core RobustMSA code (old RECAP modules are not needed),
strip the checkpoints to the keys required for inference, copy the two result
CSVs under clear names, write README / requirements, scan every file for
identity information, re-run inference from inside the package and compare it
with the submitted CSVs, then zip.
"""
import argparse
import ast
import csv
import importlib.metadata as metadata
import json
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import torch
import yaml

ROOT_NAME = '问题2_问题3_提交附件'
CODE_NAME = '代码与模型'
RESULT_NAME = '结果文件'
ATT3_CSV = '附件3_模态缺失测试集_预测结果.csv'
ATT4_CSV = '附件4_可解释测试集_预测与解释结果.csv'
CODE_FILES = [
    'train_robust.py', 'analyze_robust.py', 'predict_robust.py', 'explain_robust.py',
    'core/robust_data.py', 'core/robust_eval.py', 'models/robust_msa.py',
]
REQUIRED_MARKERS = {
    'predict_robust.py': ['bert_path', 'def missing_spans', 'RawTextEncoder'],
    'explain_robust.py': ['def normalize_fields', 'CompatUnpickler'],
}
IDENTITY_PATTERNS = [r'/home/', r'\blyz\b', r'cpz9k3', r'2829932138', r'qq\.com', r'4rtx5090',
                     r'bigmouse']
CKPT_KEYS = ('state_dict', 'config', 'normalizer', 'epoch', 'seed', 'valid_metrics', 'decision')

# ------------------------------------------------ self-contained core code
LABELS_OLD_IMPORT = 'from core.dataset import MMDataset\n'
LABELS_NEW_FUNC = '''

def normalize_classification_labels(labels, regression):
    """Map Negative/Neutral/Positive (strings, -1/0/1 or 0/1/2) to 0/1/2 and
    check that every label agrees with the sign of the regression label."""
    labels = np.asarray(labels).reshape(-1)
    regression = np.asarray(regression).reshape(-1)
    if labels.dtype.kind in {'U', 'S', 'O'}:
        mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
        normalized = np.asarray([mapping[str(v).strip().lower()] for v in labels], dtype=np.int64)
    else:
        normalized = labels.astype(np.int64)
        values = set(np.unique(normalized).tolist())
        if values.issubset({-1, 0, 1}):
            normalized = normalized + 1
        elif not values.issubset({0, 1, 2}):
            raise ValueError(f'Unexpected classification labels {sorted(values)}')
    expected = np.where(regression < 0, 0, np.where(regression > 0, 2, 1))
    mismatch = int(np.count_nonzero(normalized != expected))
    if mismatch:
        raise ValueError(f'{mismatch} classification labels disagree with regression_labels')
    return normalized
'''
LABELS_OLD_CALL = '''            classes = MMDataset._normalize_classification_labels(
                split['classification_labels'], regression
            ).reshape(-1)'''
LABELS_NEW_CALL = '''            classes = normalize_classification_labels(
                split['classification_labels'], regression
            )'''

SEED_OLD_IMPORT = 'from core.utils import setup_seed\n'
SEED_OLD_OS = 'import os\n\nimport numpy as np'
SEED_NEW_OS = 'import os\nimport random\n\nimport numpy as np'
SEED_OLD_ANCHOR = '\n\ndef parse_args():'
SEED_NEW_ANCHOR = '''

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def parse_args():'''

BERT_OLD = '''class TextBertFrontend(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        from models.bert import BertTextEncoder
        self.bert = BertTextEncoder(use_finetune=True, pretrained=cfg['bert_pretrained'])
        frozen = cfg.get('bert_frozen_layers', 8)
        for parameter in self.bert.model.embeddings.parameters():
            parameter.requires_grad = False
        for layer in self.bert.model.encoder.layer[:frozen]:
            for parameter in layer.parameters():
                parameter.requires_grad = False

    def forward(self, text_bert):
        return self.bert(text_bert.float())
'''
BERT_NEW = '''class TextBertFrontend(nn.Module):
    """Optional (text_source: bert): fine-tune the top BERT layers from text_bert."""

    def __init__(self, cfg):
        super().__init__()
        from transformers import BertModel
        self.model = BertModel.from_pretrained(cfg['bert_pretrained'])
        frozen = cfg.get('bert_frozen_layers', 8)
        for parameter in self.model.embeddings.parameters():
            parameter.requires_grad = False
        for layer in self.model.encoder.layer[:frozen]:
            for parameter in layer.parameters():
                parameter.requires_grad = False

    def forward(self, text_bert):
        ids, mask, segments = text_bert[:, 0].long(), text_bert[:, 1].float(), text_bert[:, 2].long()
        return self.model(input_ids=ids, attention_mask=mask, token_type_ids=segments)[0]
'''

PATCHES = {
    'core/robust_data.py': [(LABELS_OLD_IMPORT, ''),
                            ('\n\n\ndef clean_features(',
                             '\n' + LABELS_NEW_FUNC + '\n\ndef clean_features('),
                            (LABELS_OLD_CALL, LABELS_NEW_CALL)],
    'train_robust.py': [(SEED_OLD_IMPORT, ''), (SEED_OLD_OS, SEED_NEW_OS),
                        (SEED_OLD_ANCHOR, SEED_NEW_ANCHOR)],
    'models/robust_msa.py': [(BERT_OLD, BERT_NEW)],
}
DONE_MARKERS = {
    'core/robust_data.py': 'def normalize_classification_labels',
    'train_robust.py': 'def setup_seed',
    'models/robust_msa.py': 'from transformers import BertModel',
}


def patch_source(relative, text):
    """Remove dependencies on the old RECAP modules (idempotent)."""
    if relative not in PATCHES or DONE_MARKERS[relative] in text:
        return text
    for old, new in PATCHES[relative]:
        if text.count(old) != 1:
            raise RuntimeError(f'{relative}: expected text not found, cannot patch:\n{old[:80]}')
        text = text.replace(old, new)
    return text


# ------------------------------------------------------------- documents
def environment():
    packages = ['torch', 'numpy', 'scipy', 'scikit-learn', 'PyYAML', 'transformers',
                'opencv-python', 'opencv-python-headless', 'matplotlib']
    versions = {}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return versions


def requirements_text(versions):
    lines = [f'# Python {platform.python_version()}; CUDA (torch) {torch.version.cuda}',
             '# opencv is optional: only used to convert evidence to seconds and save keyframes']
    for name in ('torch', 'numpy', 'scipy', 'scikit-learn', 'PyYAML', 'transformers'):
        if name in versions:
            lines.append(f'{name}=={versions[name]}')
    for name in ('opencv-python', 'opencv-python-headless'):
        if name in versions:
            lines.append(f'# {name}=={versions[name]}')
    return '\n'.join(lines) + '\n'


def metric_row(name, metrics):
    return (f'| {name} | {metrics["MAE"]:.4f} | {metrics["Corr"]:.4f} | '
            f'{metrics["Polarity_Accuracy"]:.4f} | {metrics["Polarity_Macro_F1"]:.4f} |')


VIEW_NAMES = {'complete': '完整', 'missing': '缺失'}
SPLIT_NAMES = {'valid': '验证集', 'test': '测试集'}


def ensemble_metrics(log_path):
    """Parse the 'valid/test complete/missing {...}' lines of analyze_robust.py."""
    rows = []
    if log_path.exists():
        for line in log_path.read_text(encoding='utf-8', errors='ignore').splitlines():
            match = re.match(r'^(valid|test)\s+(complete|missing)\s+(\{.*\})\s*$', line)
            if match:
                rows.append((match.group(1), match.group(2), ast.literal_eval(match.group(3))))
    return rows


def readme_text(cfg, versions, checkpoints, faithfulness, ensemble):
    aug, model, optim, loss = cfg['data']['augment'], cfg['model'], cfg['optim'], cfg['loss']
    rows = [metric_row(f'三模型集成 {SPLIT_NAMES[split]}（{VIEW_NAMES[view]}）', metrics)
            for split, view, metrics in ensemble]
    rows += [metric_row(f'单模型 seed {c["seed"]} 验证集（{VIEW_NAMES[view]}）',
                        c['valid_metrics'][view])
             for c in checkpoints for view in ('complete', 'missing')]
    valid_rows = '\n'.join(rows)
    faith = ''
    if faithfulness:
        faith_rows = '\n'.join(
            f'| {m} | {faithfulness[m]["occlusion_top"]} | {faithfulness[m]["attention_top"]} | '
            f'{faithfulness[m]["random"]} | {faithfulness[m]["spearman"]} |'
            for m in ('text', 'audio', 'vision'))
        removal = faithfulness['modality_removal']
        faith = f'''
### 可解释性忠实度检验（附件2验证集，删除实验）

删除 k=3 个证据片段后预测强度的平均变化量 |Δ|（越大说明被删除的内容越关键）：

| 模态 | 删除遮挡法 top-3 | 删除注意力 top-3 | 随机删除 3 个 | Spearman(遮挡, 注意力) |
|---|---|---|---|---|
{faith_rows}

整模态删除：删除 Shapley 贡献最大的模态 |Δ|={removal["top_shapley"]}，删除其他模态 |Δ|={removal["other"]}。
'''
    env = '\n'.join(f'- {name} {version}' for name, version in versions.items())
    return f'''# 问题2 / 问题3：缺失感知的可解释多模态情感预测模型（RobustMSA）

本目录包含问题2（模态局部缺失条件下的鲁棒预测）与问题3（可解释性预测）共用的
核心代码、模型参数、配置文件与运行说明。两个问题使用同一个模型：问题2输出极性与强度，
问题3在此基础上输出模态作用程度、主要参考模态与关键证据定位。

## 1. 目录结构

```
代码与模型/
├── README.md                     本说明
├── requirements.txt              运行环境
├── configs/robust_mosei.yaml     全部超参数
├── core/robust_data.py           数据读取、缺失判定、标准化、训练缺失增强
├── core/robust_eval.py           评价指标、验证集极性决策校准、批量推理
├── models/robust_msa.py          模型结构与损失函数
├── train_robust.py               训练（附件2 train 训练、valid 选模型）
├── analyze_robust.py             三模型集成评估、集成极性规则、缺失规律分析（问题2）
├── predict_robust.py             附件3推理，输出预测结果 CSV（问题2）
├── explain_robust.py             附件4推理与解释，忠实度检验（问题3）
└── checkpoints/
    ├── robust_robust_v1_seed1111.pth   三个随机种子的模型参数（集成使用）
    ├── robust_robust_v1_seed2222.pth
    ├── robust_robust_v1_seed3333.pth
    └── ensemble_decision.json          验证集上确定的集成极性决策规则
```

## 2. 运行环境

{env}
- Python {platform.python_version()}，CUDA {torch.version.cuda}（CPU 也可运行，速度较慢）
- 文本编码使用开源预训练模型 `bert-base-uncased`（HuggingFace）。可联网时直接写
  `--bert_path bert-base-uncased`；离线时下载该模型目录后写本地路径。

安装：`pip install -r requirements.txt`（torch 请按本机 CUDA 版本安装）。

## 3. 数据与处理规则

- **数据来源**：仅使用赛题附件2的 `unaligned_50.pkl`（非对齐版本），按其原有
  train/valid/test 划分。训练只用 train，模型选择、超参数与极性决策阈值只用 valid，
  test 仅在训练结束后评估一次。附件3、附件4使用对应的"未对齐版本"，只做最终推理，
  未参与任何训练、调参或阈值选择。将附件2文件放到 `data/mosei/unaligned_50.pkl`，
  或修改配置中的 `data.path`。
- **输入特征**：文本 `text`（50×768，预计算 BERT 特征），语音 `audio`（500×74），
  视觉 `vision`（500×35）。
- **有效长度**：附件2、附件4 使用 `audio_lengths` / `vision_lengths`，文本使用
  `text_bert` 注意力掩码；附件3未提供长度字段，取"最后一个非全零行 + 1"。
- **缺失判定**：有效长度内特征值全为 0 的行判为缺失（与附件3的缺失定义一致）。
  缺失位置由可学习的"缺失向量"替代，不参与时序注意力池化，因而不会被报告为关键证据。
  附件2测试集语音有效长度内全零行比例为 0，视觉为约 1% 样本含少量零行，
  说明附件3中的全零行基本为人为缺失。
- **数值清洗与标准化**：nan / ±inf 置 0；三个模态分别用训练集"非缺失行"的逐维
  均值与标准差做 z-score，并裁剪到 ±10。统计量保存在模型参数文件中，推理时直接使用。
- **附件3文本特征重建**：附件3只提供 `raw_text`，没有 `text` 特征。使用
  `bert-base-uncased` 对 `raw_text` 分词（最大长度 50）并取最后一层隐藏状态，
  与附件2的生成方式一致。在附件2验证集上核对：分词结果与 `text_bert` 100% 一致，
  特征与 `text` 的平均余弦相似度 1.0000。
- **附件4读取**：附件4为 NumPy 2.x 格式的 pickle，字段不带样本维；代码兼容读取并补齐样本维。
- **训练缺失增强（仅训练集）**：每个模态以 {aug["span_prob"]} 的概率置零 1–{aug["max_spans"]} 个
  连续片段，合计占有效长度的 {aug["min_ratio"]:.0%}–{aug["max_ratio"]:.0%}；并以
  {aug["modality_drop_prob"]} 的概率将某一整个模态置零（不会三个模态同时置零）。
- 未删除、替换或修改任何样本及标签。

## 4. 模型与训练方案

- **模态编码**：每个模态一个 {model["layers"]} 层 Transformer 编码器（隐藏维 {model["hidden_dim"]}，
  {model["heads"]} 头）。语音、视觉的 500 帧先按 5 帧窗口做"仅对非缺失帧"的平均池化，得到 100 步。
- **可靠性门控融合**：门控网络输入每个模态的池化向量、非缺失比例与可用标志，
  softmax 输出三模态权重；缺失多的模态自动降权。
- **输出**：强度回归头（3·tanh(x/3)，范围 [-3, 3]）、三分类极性头、三个单模态辅助强度头。
- **损失**：MAE + {loss["cls"]}×类别加权交叉熵（标签平滑 {loss["label_smoothing"]}）
  + {loss["unimodal"]}×单模态 MAE + {loss["corr"]}×(1 − Pearson r)。
- **优化**：AdamW，学习率 {optim["lr"]}，权重衰减 {optim["weight_decay"]}，
  {optim["warmup_epochs"]} 轮预热 + 余弦退火，最多 {optim["epochs"]} 轮，早停耐心
  {optim["early_stopping_patience"]} 轮，梯度裁剪 {optim["grad_clip"]}，批大小 {cfg["data"]["batch_size"]}。
- **模型选择**：每轮在验证集的"完整"与"固定种子连续缺失"两种版本上计算
  Macro-F1 + Pearson r − MAE，取两者平均最高的轮次。
- **极性决策**：训练结束后在验证集上搜索类别对数偏置或强度阈值，取 Macro-F1 最高的规则；
  三模型集成的规则由 `analyze_robust.py` 在验证集上重新确定，保存为 `ensemble_decision.json`。
- **集成**：3 个随机种子（1111 / 2222 / 3333）的模型输出取平均。

## 5. 可解释性方法（问题3）

- **模态作用程度**：对三个模态的全部 8 种组合（被移除的模态置零，训练时模型见过整模态缺失）
  计算精确 Shapley 值。满足可加性：基线（三模态全移除）+ 三个贡献之和 = 完整预测，
  CSV 中 `additivity_error` 为该等式的误差。
- **主要参考模态**：|Shapley 值| 最大的模态；门控权重最大的模态另列 `main_modality_gate` 作对照。
- **关键证据定位**：逐个置零文本词片或语音/视觉的 5 帧窗口，以预测强度的变化量作为重要性
  （正值表示该片段把预测推向正向）。每个模态报告前 3 个证据：文本给出词与位置；
  语音/视觉给出帧区间，若能读取对应视频则换算为秒（秒 = 帧 / 有效长度 × 视频时长，
  假设特征帧在片段内均匀分布）。
- **可复核性**：`explain_robust.py --faithfulness_samples N` 在附件2验证集上做删除实验，
  比较删除报告的证据与随机删除对预测的影响。

## 6. 复现步骤（在本目录下运行）

### 6.1 用提供的模型直接复现提交结果

```bash
# 问题2：附件3预测
python predict_robust.py --checkpoints "checkpoints/robust_robust_v1_seed*.pth" \\
  --decision_json checkpoints/ensemble_decision.json --bert_path bert-base-uncased \\
  --input_dir <附件3目录>/未对齐版本 --output_csv outputs/附件3_预测结果.csv

# 问题3：附件4预测与解释（视频目录默认为 <input_dir>/videos）
python explain_robust.py --checkpoints "checkpoints/robust_robust_v1_seed*.pth" \\
  --decision_json checkpoints/ensemble_decision.json --bert_path bert-base-uncased \\
  --input_dir <附件4目录>/未对齐版本 --output_csv outputs/附件4_预测与解释结果.csv
```

输出与 `结果文件/` 中的两个 CSV 一致（打包时已在本目录下重新运行并逐行核对）。

### 6.2 从头训练并复现分析

```bash
for s in 1111 2222 3333; do python train_robust.py --config_file configs/robust_mosei.yaml --seed $s; done
python analyze_robust.py --checkpoints "checkpoints_retrained/robust_robust_v1_seed*.pth"
python explain_robust.py --checkpoints "checkpoints_retrained/robust_robust_v1_seed*.pth" \\
  --decision_json outputs/ensemble_decision.json --bert_path bert-base-uncased \\
  --faithfulness_samples 500
```

- 新模型保存在 `checkpoints_retrained/`，不会覆盖提供的模型。
- `analyze_robust.py` 输出集成指标、`outputs/ensemble_decision.json`（集成极性规则）和
  `outputs/missing_analysis.csv`（缺失模态组合 × 缺失位置 开头/中间/结尾/随机 ×
  缺失时长 10%–100% 的性能，用于问题2的规律分析）。
- 训练中的缺失增强为在线随机采样，重新训练的指标会与下表有小幅差异；
  使用提供的模型参数可以精确复现提交的 CSV。

## 7. 结果文件字段

**附件3（`{ATT3_CSV}`）**：`sample_id` 样本编号；`polarity_pred` 情感极性
（Negative / Neutral / Positive）；`intensity_pred` 情感强度 [-3, 3]；`prob_*` 三类概率；
`main_modality` 门控权重最大的模态；每个模态的 `*_weight` 门控权重、`*_intensity`
单模态强度、`*_observed_ratio` 逐帧非缺失比例、`*_missing_frames` 缺失帧数、
`*_missing_spans` 缺失区间（帧，含端点）、`*_gate_reliability` 门控使用的 5 帧窗口可靠度、
`*_length` 有效长度、`*_evidence` 注意力权重最高的片段。

**附件4（`{ATT4_CSV}`）**：`sample_id` / `video_file` / `video_duration_s`；
`polarity_pred`、`intensity_pred`、`prob_*`；`main_modality` 主要参考模态（Shapley）；
`main_modality_gate` 门控最大模态；`baseline_intensity` 三模态全移除时的预测；
`additivity_error` Shapley 可加性误差；每个模态的 `*_shapley` 贡献值、
`*_contribution_share` 贡献占比、`*_gate_weight`、`*_unimodal_intensity`、
`*_observed_ratio`、`*_evidence` 前 3 个关键证据（文本："词"(token 位置, 相对位置, 影响)；
语音/视觉：帧区间 | 秒区间 (影响)）；`vision_keyframe_s` 视觉首要证据的时间点（秒，若可读视频）。

## 8. 参考结果（附件2）

| 集合 | MAE ↓ | Pearson r ↑ | 极性准确率 ↑ | 极性 Macro-F1 ↑ |
|---|---|---|---|---|
{valid_rows}

"完整"为原始数据，"缺失"为按训练增强规则固定随机种子生成的连续缺失版本。
集成结果由 `analyze_robust.py` 输出；测试集只在模型与决策规则确定后评估一次。
{faith}'''


# ---------------------------------------------------------------- helpers
def read_rows(path):
    with open(path, encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows, drop=()):
    fields = [key for key in rows[0] if key not in drop]
    with open(path, 'w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def scan_identity(root):
    hits = []
    for path in sorted(root.rglob('*')):
        if not path.is_file():
            continue
        if path.suffix == '.pth':
            checkpoint = torch.load(path, map_location='cpu')
            text = json.dumps({k: v for k, v in checkpoint.items() if k != 'state_dict'},
                              default=str, ensure_ascii=False)
        else:
            text = path.read_text(encoding='utf-8-sig', errors='ignore')
        for pattern in IDENTITY_PATTERNS:
            for match in re.finditer(pattern, text):
                start = max(0, match.start() - 40)
                hits.append(f'{path.relative_to(root)}: ...{text[start:match.end() + 40]}...')
    return hits


def compare(submitted, rerun, keys, tolerance=1e-3):
    a, b = read_rows(submitted), read_rows(rerun)
    if len(a) != len(b):
        return [f'row count {len(a)} vs {len(b)}']
    problems = []
    for row_a, row_b in zip(a, b):
        for key in keys:
            if key not in row_a:
                continue
            x, y = row_a[key], row_b[key]
            try:
                same = abs(float(x) - float(y)) <= tolerance
            except ValueError:
                same = x == y
            if not same:
                problems.append(f'{row_a["sample_id"]} {key}: {x} vs {y}')
    return problems


def run(command, cwd):
    print('  $ ' + ' '.join(command))
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-3000:])
        print(result.stderr[-3000:])
        raise RuntimeError('verification command failed')


# ------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', default='.')
    parser.add_argument('--checkpoints', default='ckpt/robust_mosei/robust_robust_v1_seed*.pth')
    parser.add_argument('--decision_json', default='outputs/ensemble_decision.json')
    parser.add_argument('--att3_csv', default='outputs/attachment3_predictions.csv')
    parser.add_argument('--att4_csv', default='outputs/attachment4_explanations.csv')
    parser.add_argument('--faithfulness_json', default='outputs/faithfulness.json')
    parser.add_argument('--analyze_log', default='logs/robust_analyze.log')
    parser.add_argument('--att3_dir', required=True)
    parser.add_argument('--att4_dir', required=True)
    parser.add_argument('--bert_path', default='bert-base-uncased')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output_dir', default='submission')
    parser.add_argument('--skip_verify', action='store_true')
    parser.add_argument('--patch_repo', action='store_true',
                        help='only apply the self-contained patches to the repo sources')
    args = parser.parse_args()
    repo = Path(args.repo).resolve()

    if args.patch_repo:
        for relative in PATCHES:
            path = repo / relative
            path.write_text(patch_source(relative, path.read_text(encoding='utf-8')), encoding='utf-8')
            print(f'patched {relative}')
        return

    for relative, markers in REQUIRED_MARKERS.items():
        text = (repo / relative).read_text(encoding='utf-8')
        missing = [m for m in markers if m not in text]
        if missing:
            raise SystemExit(f'{relative} is an old version (missing {missing}); update it first')

    out = Path(args.output_dir).resolve() / ROOT_NAME
    if out.exists():
        shutil.rmtree(out)
    code, results = out / CODE_NAME, out / RESULT_NAME
    print(f'[1/6] core code -> {code}')
    for relative in CODE_FILES:
        target = code / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        text = patch_source(relative, (repo / relative).read_text(encoding='utf-8'))
        target.write_text(text, encoding='utf-8')
    for package in ('core', 'models'):
        (code / package / '__init__.py').write_text('', encoding='utf-8')

    print('[2/6] checkpoints (inference keys only)')
    ckpt_dir = code / 'checkpoints'
    ckpt_dir.mkdir()
    checkpoints = []
    for path in sorted(repo.glob(args.checkpoints)):
        checkpoint = torch.load(path, map_location='cpu')
        slim = {key: checkpoint[key] for key in CKPT_KEYS if key in checkpoint}
        slim['config']['checkpoint_dir'] = 'checkpoints_retrained'
        slim['config']['data']['path'] = 'data/mosei/unaligned_50.pkl'
        slim['config']['model']['bert_pretrained'] = 'bert-base-uncased'
        torch.save(slim, ckpt_dir / path.name)
        checkpoints.append(slim)
        print(f'  {path.name}: seed {slim["seed"]}, epoch {slim["epoch"]}')
    if len(checkpoints) != 3:
        raise SystemExit(f'expected 3 checkpoints, found {len(checkpoints)}')
    shutil.copy(repo / args.decision_json, ckpt_dir / 'ensemble_decision.json')
    cfg = checkpoints[0]['config']
    (code / 'configs').mkdir()
    with open(code / 'configs' / 'robust_mosei.yaml', 'w', encoding='utf-8') as handle:
        yaml.safe_dump(cfg, handle, allow_unicode=True, sort_keys=False)

    print('[3/6] result CSVs')
    results.mkdir()
    att3, att4 = read_rows(repo / args.att3_csv), read_rows(repo / args.att4_csv)
    print(f'  Attachment 3: {len(att3)} rows; Attachment 4: {len(att4)} rows')
    if len(att3) != 30 or len(att4) != 20:
        print('  WARNING: expected 30 and 20 rows; check the input directories')
    write_rows(results / ATT3_CSV, att3)
    write_rows(results / ATT4_CSV, att4, drop=('vision_keyframe',))

    print('[4/6] README / requirements')
    versions = environment()
    faithfulness = None
    if (repo / args.faithfulness_json).exists():
        faithfulness = json.loads((repo / args.faithfulness_json).read_text(encoding='utf-8'))
    (code / 'requirements.txt').write_text(requirements_text(versions), encoding='utf-8')
    (code / 'README.md').write_text(readme_text(cfg, versions, checkpoints, faithfulness,
                                                ensemble_metrics(repo / args.analyze_log)),
                                    encoding='utf-8')

    print('[5/6] verification')
    run([sys.executable, '-c',
         'import train_robust, analyze_robust, predict_robust, explain_robust; print("imports ok")'],
        cwd=code)
    if not args.skip_verify:
        with tempfile.TemporaryDirectory() as tmp:
            common = ['--checkpoints', 'checkpoints/robust_robust_v1_seed*.pth',
                      '--decision_json', 'checkpoints/ensemble_decision.json',
                      '--bert_path', args.bert_path, '--device', args.device]
            run([sys.executable, 'predict_robust.py', *common, '--input_dir', args.att3_dir,
                 '--output_csv', f'{tmp}/att3.csv'], cwd=code)
            run([sys.executable, 'explain_robust.py', *common, '--input_dir', args.att4_dir,
                 '--output_csv', f'{tmp}/att4.csv', '--keyframe_dir', ''], cwd=code)
            problems = compare(results / ATT3_CSV, f'{tmp}/att3.csv',
                               ['sample_id', 'polarity_pred', 'intensity_pred'])
            problems += compare(results / ATT4_CSV, f'{tmp}/att4.csv',
                                ['sample_id', 'polarity_pred', 'intensity_pred', 'main_modality',
                                 'text_shapley', 'audio_shapley', 'vision_shapley'])
        if problems:
            print('\n'.join(problems[:20]))
            raise SystemExit('re-run from the package does NOT match the submitted CSVs')
        print('  re-run from the package matches both submitted CSVs')
    for path in code.rglob('__pycache__'):
        shutil.rmtree(path)

    print('[6/6] identity scan and zip')
    hits = scan_identity(out)
    if hits:
        print('\n'.join(hits[:30]))
        raise SystemExit('identity information found; fix the files above before submitting')
    archive = out.parent / f'{ROOT_NAME}.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(out.rglob('*')):
            if path.is_file():
                handle.write(path, path.relative_to(out.parent))
    size = archive.stat().st_size / 2 ** 20
    print(f'\n{archive}  {size:.1f} MB')
    for path in sorted(out.rglob('*')):
        if path.is_file():
            print(f'  {path.relative_to(out.parent)}  {path.stat().st_size / 1024:.0f} KB')
    if size > 50:
        raise SystemExit('archive exceeds 50 MB')


if __name__ == '__main__':
    main()
