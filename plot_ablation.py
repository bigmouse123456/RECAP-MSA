"""Figures for the ablation study and parameter sensitivity (PNG, 200 dpi).

    python plot_ablation.py                 # validation curves (hyper-parameters are chosen on valid)
    python plot_ablation.py --split test

Reads outputs/ablation/param_summary.csv and ablation_summary.csv written by
run_ablation.py and saves to outputs/figures/:
    p2_param_sensitivity_<split>.png   line charts, one column per key parameter
    p2_ablation_delta.png              change of each ablation vs the full model (test)
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from plot_explanations import GRID, INK, INK2, SURFACE, save, style  # noqa: E402

VIEW_STYLE = {'complete': ('Complete input', '#2a78d6'), 'missing': ('Contiguous missing', '#eb6834')}
ROWS = [('MAE', 'MAE (lower is better)'), ('Corr', 'Pearson r'),
        ('Polarity_Macro_F1', 'Polarity macro-F1')]
PARAM_ORDER = ['max_ratio', 'cls_weight', 'hidden_dim', 'stride']


def read_csv(path):
    with open(path, encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def value_label(value):
    number = float(value)
    return str(int(number)) if number.is_integer() and number >= 1 else f'{number:g}'


def param_figure(rows, split, path):
    rows = [r for r in rows if r['split'] == split]
    params = [p for p in PARAM_ORDER if any(r['param'] == p for r in rows)]
    fig, axes = plt.subplots(len(ROWS), len(params), figsize=(3.3 * len(params), 7.6),
                             squeeze=False)
    for col, param in enumerate(params):
        points = sorted((r for r in rows if r['param'] == param), key=lambda r: float(r['value']))
        x = np.arange(len(points))
        default = [i for i, r in enumerate(points) if r['is_default'] == 'True']
        for row_index, (metric, ylabel) in enumerate(ROWS):
            ax = axes[row_index][col]
            for view, (label, color) in VIEW_STYLE.items():
                mean = np.array([float(r[f'{view}_{metric}_mean']) for r in points])
                std = np.array([float(r[f'{view}_{metric}_std']) for r in points])
                ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)
                ax.plot(x, mean, color=color, marker='o', markersize=5, label=label)
            for i in default:
                ax.axvline(i, color=INK2, linewidth=0.8, zorder=0)
                if row_index == 0:
                    ax.annotate('default', (i, 1.0), xycoords=('data', 'axes fraction'),
                                xytext=(3, -10), textcoords='offset points', fontsize=7, color=INK2)
            ax.set_xticks(x)
            ax.set_xticklabels([value_label(r['value']) for r in points])
            ax.set_xlim(-0.3, len(points) - 0.7)
            ax.grid(axis='x', visible=False)
            if col == 0:
                ax.set_ylabel(ylabel)
            if row_index == 0:
                ax.set_title(points[0]['label_en'], fontsize=9)
            if row_index == len(ROWS) - 1:
                ax.set_xlabel('value')
    n = points[0]['n_seeds'] if params else '?'
    fig.suptitle(f'Key-parameter sensitivity on Attachment 2 {split} '
                 f'(mean ± std over {n} seeds; other parameters at default)',
                 x=0.01, ha='left', fontsize=11, fontweight='bold', y=0.995)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.01, 0.955), ncol=2,
               fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, path)


def ablation_figure(rows, path):
    rows = [r for r in rows if r['split'] == 'test']
    full = next(r for r in rows if r['variant'] == 'full')
    others = [r for r in rows if r['variant'] != 'full']
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.42 * len(others) + 1.8), sharey=True)
    panels = [('MAE', 'Δ MAE vs full model (positive = worse)'),
              ('Polarity_Macro_F1', 'Δ macro-F1 vs full model (negative = worse)')]
    y = np.arange(len(others))[::-1]
    height = 0.36
    for ax, (metric, title) in zip(axes, panels):
        for offset, (view, (label, color)) in zip((height / 2 + 0.02, -height / 2 - 0.02),
                                                  VIEW_STYLE.items()):
            delta = np.array([float(r[f'{view}_{metric}_mean']) - float(full[f'{view}_{metric}_mean'])
                              for r in others])
            ax.barh(y + offset, delta, height=height, color=color, label=label)
            for yi, value in zip(y + offset, delta):
                ax.annotate(f'{value:+.3f}', (value, yi), xytext=(3 if value >= 0 else -3, 0),
                            textcoords='offset points', va='center', fontsize=7, color=INK,
                            ha='left' if value >= 0 else 'right')
        ax.axvline(0, color=INK2, linewidth=0.8)
        span = max(abs(v) for v in ax.get_xlim()) * 1.25
        ax.set_xlim(-span, span)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5, symmetric=True))
        ax.grid(axis='y', visible=False)
        ax.set_title(title, fontsize=9)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([r['label_en'] for r in others], fontsize=8)
    fig.suptitle(f'Ablation on Attachment 2 test (full model: complete MAE '
                 f'{float(full["complete_MAE_mean"]):.3f}, F1 {float(full["complete_Polarity_Macro_F1_mean"]):.3f}; '
                 f'missing MAE {float(full["missing_MAE_mean"]):.3f}, '
                 f'F1 {float(full["missing_Polarity_Macro_F1_mean"]):.3f})',
                 x=0.01, ha='left', fontsize=10, fontweight='bold', y=0.995)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.01, 0.955), ncol=2,
               fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='outputs/ablation')
    parser.add_argument('--output_dir', default='outputs/figures')
    parser.add_argument('--split', default='valid', choices=['valid', 'test'])
    args = parser.parse_args()
    style()
    source, out = Path(args.input_dir), Path(args.output_dir)
    if (source / 'param_summary.csv').exists():
        param_figure(read_csv(source / 'param_summary.csv'), args.split,
                     out / f'p2_param_sensitivity_{args.split}.png')
    if (source / 'ablation_summary.csv').exists():
        ablation_figure(read_csv(source / 'ablation_summary.csv'), out / 'p2_ablation_delta.png')


if __name__ == '__main__':
    main()
