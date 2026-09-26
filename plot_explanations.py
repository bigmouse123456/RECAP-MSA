"""Static report figures for Problems 2 and 3 (PNG, 200 dpi).

Reads the files written by analyze_robust.py and explain_robust.py:

    python plot_explanations.py            # all figures -> outputs/figures/
    python plot_explanations.py --samples 04 09 05

Problem 3: an explanation card per typical sample, modality contribution
shares for every Attachment 4 sample, the within-modality position profile of
evidence importance, and the deletion (faithfulness) test.
Problem 2: metrics vs missing ratio and vs missing position.

Colors follow the entity: text / audio / vision keep one hue in every figure.
The data behind each figure is in the CSV / JSON files it reads.
"""
import argparse
import csv
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

MODALITIES = ('text', 'audio', 'vision')
LABELS = {'text': 'Text', 'audio': 'Audio', 'vision': 'Vision'}
COLORS = {'text': '#2a78d6', 'audio': '#eb6834', 'vision': '#1baf7a'}
ALL_MISSING = '#52514e'
METHODS = {  # validated as a set (all pairs, light surface)
    'occlusion_top': ('Top-k by occlusion', '#4a3aa7'),
    'attention_top': ('Top-k by attention', '#e87ba4'),
    'random': ('Random k', '#eda100'),
}
SURFACE, INK, INK2, GRID, MISSING_FILL = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df', '#ecebe7'


def style():
    plt.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
        'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'axes.titlecolor': INK,
        'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
        'axes.spines.top': False, 'axes.spines.right': False,
        'font.size': 9, 'axes.titlesize': 10, 'axes.titleweight': 'bold',
        'axes.titlelocation': 'left', 'legend.frameon': False, 'lines.linewidth': 2,
    })


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {path}')


def read_csv(path):
    with open(path, encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def zero_line(ax, orientation='h'):
    (ax.axhline if orientation == 'h' else ax.axvline)(0, color=INK2, linewidth=0.8)


# ------------------------------------------------------------- Problem 3
def pick_samples(rows):
    """Conflicting modalities, strongest text-driven, strongest non-text-driven."""
    def phi(row):
        return np.array([float(row[f'{m}_shapley']) for m in MODALITIES])

    picks = []
    conflict = [r for r in rows if phi(r).max() > 0 > phi(r).min()]
    if conflict:
        picks.append(max(conflict, key=lambda r: min(phi(r).max(), -phi(r).min())))
    text_main = [r for r in rows if r['main_modality'] == 'text']
    if text_main:
        picks.append(max(text_main, key=lambda r: abs(float(r['text_shapley']))))
    other = [r for r in rows if r['main_modality'] != 'text']
    if other:
        picks.append(max(other, key=lambda r: abs(float(r[f'{r["main_modality"]}_shapley']))))
    seen, unique = set(), []
    for row in picks:
        if row['sample_id'] not in seen:
            seen.add(row['sample_id'])
            unique.append(row['sample_id'])
    return unique


def temporal_panel(ax, modality, detail, top_k):
    info = detail['modalities'][modality]
    windows = np.asarray(info['windows'], dtype=float).reshape(-1, 2)
    effect = np.asarray(info['occlusion'], dtype=float)
    length = detail['lengths'][modality]
    duration = detail.get('duration')
    color = COLORS[modality]
    if not len(effect):
        ax.text(0.5, 0.5, 'no observed positions', ha='center', va='center',
                transform=ax.transAxes, color=INK2)
        return
    top = set(np.argsort(-np.abs(effect))[:top_k].tolist())
    alpha = [1.0 if i in top else 0.4 for i in range(len(effect))]

    if modality == 'text':
        tokens = detail.get('tokens') or []
        x = np.arange(len(effect))
        ax.bar(x, effect, width=0.8, color=color, alpha=None)
        for bar, a in zip(ax.patches, alpha):
            bar.set_alpha(a)
        names = [tokens[int(a)] if int(a) < len(tokens) else str(int(a)) for a, _ in windows]
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=60, ha='right', fontsize=7)
        ax.set_xlim(-0.8, len(x) - 0.2)
        ax.grid(axis='x', visible=False)
        for i in top:
            ax.annotate(f'{effect[i]:+.2f}', (x[i], effect[i]), xytext=(0, 3 if effect[i] >= 0 else -9),
                        textcoords='offset points', ha='center', fontsize=7, color=INK)
        ax.set_title(f'{LABELS[modality]}: effect of removing each word piece')
    else:
        scale = duration / length if duration else 1.0
        centers = windows.mean(axis=1) * scale
        widths = (windows[:, 1] - windows[:, 0]) * scale * 0.85
        for start, end in info.get('missing_spans', []):
            ax.axvspan(start * scale, (end + 1) * scale, color=MISSING_FILL, linewidth=0, zorder=0)
        bars = ax.bar(centers, effect, width=widths, color=color)
        for bar, a in zip(bars, alpha):
            bar.set_alpha(a)
        unit = 's' if duration else ''
        for i in top:
            a, b = windows[i] * scale
            label = f'{a:.2f}-{b:.2f}s' if duration else f'frames {int(windows[i][0])}-{int(windows[i][1]) - 1}'
            ax.annotate(label, (centers[i], effect[i]), xytext=(0, 3 if effect[i] >= 0 else -9),
                        textcoords='offset points', ha='center', fontsize=7, color=INK)
        ax.set_xlim(0, length * scale)
        ax.set_xlabel('Time in clip (s)' if unit else 'Frame index')
        ax.set_title(f'{LABELS[modality]}: effect of removing each '
                     f'{int(windows[0][1] - windows[0][0])}-frame window')
    zero_line(ax)
    ax.set_ylabel('Δ intensity\n(+ pushes positive)')
    pad = np.abs(effect).max() * 0.25 or 0.01
    ax.set_ylim(min(effect.min(), 0) - pad, max(effect.max(), 0) + pad)


def explanation_card(row, detail, top_k, path):
    fig = plt.figure(figsize=(11, 10))
    grid = fig.add_gridspec(4, 3, height_ratios=[1.0, 1.25, 1.0, 1.0],
                            width_ratios=[1, 1, 0.75], hspace=0.95, wspace=0.35)
    fig.text(0.07, 0.975, f'Sample {row["sample_id"]}: {row["polarity_pred"]}, intensity '
             f'{float(row["intensity_pred"]):+.2f}; main modality: {LABELS[row["main_modality"]]}',
             fontsize=13, fontweight='bold', va='top')
    fig.text(0.07, 0.945, '\n'.join(textwrap.wrap(f'"{row["raw_text"]}"', 140)[:3]),
             fontsize=9, color=INK2, va='top')

    ax = fig.add_subplot(grid[0, :2])
    phi = np.array([float(row[f'{m}_shapley']) for m in MODALITIES])
    share = [float(row[f'{m}_contribution_share']) for m in MODALITIES]
    y = np.arange(3)[::-1]
    ax.barh(y, phi, height=0.55, color=[COLORS[m] for m in MODALITIES])
    for yi, value, s in zip(y, phi, share):
        ax.annotate(f'{value:+.3f}  ({s:.0%})', (value, yi), xytext=(4 if value >= 0 else -4, 0),
                    textcoords='offset points', va='center',
                    ha='left' if value >= 0 else 'right', fontsize=8, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([LABELS[m] for m in MODALITIES])
    zero_line(ax, 'v')
    span = np.abs(phi).max() * 1.6 or 0.1
    ax.set_xlim(-span, span)
    ax.grid(axis='y', visible=False)
    ax.set_xlabel('Shapley contribution to intensity (share of |contribution|)')
    ax.set_title(f'Modality contributions: baseline {float(row["baseline_intensity"]):+.2f} '
                 f'+ contributions = prediction {float(row["intensity_pred"]):+.2f}')

    ax_frame = fig.add_subplot(grid[0, 2])
    ax_frame.axis('off')
    keyframe = row.get('vision_keyframe', '')
    if keyframe and Path(keyframe).exists():
        try:
            ax_frame.imshow(plt.imread(keyframe))
            ax_frame.set_title(f'Top vision evidence frame ({float(row["vision_keyframe_s"]):.2f}s)',
                               fontsize=9)
        except Exception:
            keyframe = ''
    if not keyframe:
        ax_frame.text(0.5, 0.5, 'keyframe not available', ha='center', va='center', color=INK2)

    for r, modality in enumerate(MODALITIES, start=1):
        temporal_panel(fig.add_subplot(grid[r, :]), modality, detail, top_k)
    fig.text(0.07, 0.035, 'Bars: prediction change when the piece is zeroed (full minus occluded); '
             f'dark bars are the top-{top_k} evidence; gray bands mark missing input.',
             fontsize=8, color=INK2)
    save(fig, path)


def contribution_shares(rows, path):
    rows = sorted(rows, key=lambda r: float(r['text_contribution_share']))
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(rows) + 1.6))
    y = np.arange(len(rows))
    left = np.zeros(len(rows))
    for modality in MODALITIES:
        share = np.array([float(r[f'{modality}_contribution_share']) for r in rows])
        ax.barh(y, share, left=left, height=0.7, color=COLORS[modality],
                edgecolor=SURFACE, linewidth=1.5, label=LABELS[modality])
        left += share
    for yi, row in zip(y, rows):
        ax.text(1.02, yi, f'{row["polarity_pred"]} {float(row["intensity_pred"]):+.2f}',
                va='center', fontsize=8, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([r['sample_id'] for r in rows], fontsize=8)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis='y', visible=False)
    ax.set_xlabel('Share of |Shapley contribution|')
    ax.set_title('Attachment 4: modality contribution per sample (prediction at right)', pad=22)
    ax.legend(ncol=3, loc='lower left', bbox_to_anchor=(0, 1.0), handlelength=1.2)
    save(fig, path)


def position_profile(details, path, bins=10):
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    centers = (np.arange(bins) + 0.5) / bins
    ends = {}
    for modality in MODALITIES:
        profiles = []
        for detail in details:
            info = detail['modalities'][modality]
            effect = np.abs(np.asarray(info['occlusion'], dtype=float))
            if not len(effect) or effect.sum() == 0:
                continue
            windows = np.asarray(info['windows'], dtype=float).reshape(-1, 2)
            position = windows.mean(axis=1) / detail['lengths'][modality]
            index = np.clip((position * bins).astype(int), 0, bins - 1)
            profiles.append(np.bincount(index, weights=effect / effect.sum(), minlength=bins))
        if not profiles:
            continue
        mean = np.mean(profiles, axis=0)
        ax.plot(centers, mean, color=COLORS[modality], marker='o', markersize=5,
                label=LABELS[modality])
        ends[modality] = mean[-1]
    # End labels, pushed apart so they never overlap.
    top = max(ax.get_ylim()[1], 1.3 / bins)
    gap = 0.06 * top
    placed = []
    for modality, value in sorted(ends.items(), key=lambda item: item[1]):
        y = max(value, placed[-1] + gap) if placed else value
        placed.append(y)
        ax.annotate(LABELS[modality], (centers[-1], value), xytext=(centers[-1] + 0.03, y),
                    textcoords='data', va='center', fontsize=8, color=INK)
    ax.axhline(1 / bins, color=INK2, linewidth=0.8)
    ax.annotate('uniform', (0.0, 1 / bins), xytext=(2, 3), textcoords='offset points',
                fontsize=7, color=INK2)
    ax.set_xlim(0, 1.1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Relative position in the clip (10% segments)')
    ax.set_ylabel('Share of occlusion importance')
    ax.set_title(f'Where the evidence lies inside each modality ({len(details)} Attachment 4 samples)',
                 pad=22)
    ax.legend(ncol=3, loc='lower left', bbox_to_anchor=(0, 1.0), handlelength=1.5)
    save(fig, path)


def faithfulness_chart(summary, path):
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4), gridspec_kw={'width_ratios': [1, 1, 1, 0.8]})
    for ax, modality in zip(axes, MODALITIES):
        stats = summary[modality]
        names = [m for m in METHODS if stats.get(m) is not None]
        values = [stats[m] for m in names]
        x = np.arange(len(names))
        ax.bar(x, values, width=0.6, color=[METHODS[m][1] for m in names])
        for xi, value in zip(x, values):
            ax.annotate(f'{value:.3f}', (xi, value), xytext=(0, 3), textcoords='offset points',
                        ha='center', fontsize=8, color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels([METHODS[m][0].replace(' by ', '\nby ') for m in names], fontsize=7)
        ax.grid(axis='x', visible=False)
        ax.set_ylim(0, max(values) * 1.25)
        rho = stats.get('spearman')
        ax.set_title(f'{LABELS[modality]} (Spearman {rho:.2f})' if rho is not None else LABELS[modality])
    axes[0].set_ylabel('Mean |Δ intensity| after deletion')
    removal = summary['modality_removal']
    ax = axes[3]
    values = [removal['top_shapley'], removal['other']]
    ax.bar([0, 1], values, width=0.6, color=[METHODS['occlusion_top'][1], METHODS['random'][1]])
    for xi, value in enumerate(values):
        ax.annotate(f'{value:.3f}', (xi, value), xytext=(0, 3), textcoords='offset points',
                    ha='center', fontsize=8, color=INK)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Top-Shapley\nmodality', 'Other\nmodality'], fontsize=7)
    ax.grid(axis='x', visible=False)
    ax.set_ylim(0, max(values) * 1.25)
    ax.set_title('Whole-modality removal')
    fig.suptitle('Deletion test on Attachment 2 valid: removing the reported evidence changes the '
                 'prediction more than removing random input', x=0.06, ha='left', fontsize=11,
                 fontweight='bold', y=1.04)
    save(fig, path)


# ------------------------------------------------------------- Problem 2
def missing_curves(rows, path):
    metrics = [('MAE', 'MAE (lower is better)'), ('Corr', 'Pearson r'),
               ('Polarity_Macro_F1', 'Polarity macro-F1')]
    series = [(m, LABELS[m] + ' missing', COLORS[m]) for m in MODALITIES]
    series.append(('text+audio+vision', 'All three missing', ALL_MISSING))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, (metric, title) in zip(axes, metrics):
        for key, label, color in series:
            points = sorted(
                (float(r['ratio']), float(r[metric])) for r in rows
                if r['missing_modalities'] == key and r['position'] in ('random', 'whole'))
            if not points:
                continue
            x, y = zip(*points)
            ax.plot(x, y, color=color, marker='o', markersize=5, label=label)
        ax.set_xlabel('Missing duration (share of valid length)')
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_title(title)
    axes[0].legend(loc='upper left', fontsize=8)
    fig.suptitle('Attachment 2 test: effect of missing modality and duration '
                 '(random position; 100% = whole modality)', x=0.06, ha='left',
                 fontsize=11, fontweight='bold', y=1.03)
    save(fig, path)


def missing_positions(rows, path, ratio=0.5):
    positions = ('begin', 'middle', 'end', 'random')
    metrics = [('MAE', 'MAE (lower is better)'), ('Polarity_Macro_F1', 'Polarity macro-F1')]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    for ax, (metric, title) in zip(axes, metrics):
        for offset, modality in zip((-0.18, 0.0, 0.18), MODALITIES):
            values = {r['position']: float(r[metric]) for r in rows
                      if r['missing_modalities'] == modality and abs(float(r['ratio']) - ratio) < 1e-6}
            y = [i + offset for i, p in enumerate(positions) if p in values]
            x = [values[p] for p in positions if p in values]
            ax.scatter(x, y, s=40, color=COLORS[modality], edgecolor=SURFACE, linewidth=1.5,
                       label=f'{LABELS[modality]} missing', zorder=3)
        ax.set_yticks(range(len(positions)))
        ax.set_yticklabels(positions)
        ax.invert_yaxis()
        ax.grid(axis='y', visible=False)
        ax.set_title(title)
    axes[0].set_ylabel('Position of the missing span')
    axes[0].legend(loc='lower left', bbox_to_anchor=(0, 1.08), ncol=3, fontsize=8)
    fig.suptitle(f'Attachment 2 test: effect of missing position ({ratio:.0%} of the valid length missing)',
                 x=0.06, ha='left', fontsize=11, fontweight='bold', y=1.12)
    save(fig, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--explanations_csv', default='outputs/attachment4_explanations.csv')
    parser.add_argument('--explanations_json', default='')
    parser.add_argument('--faithfulness_json', default='outputs/faithfulness.json')
    parser.add_argument('--missing_csv', default='outputs/missing_analysis.csv')
    parser.add_argument('--samples', nargs='*', default=None,
                        help='sample ids for explanation cards (default: automatic)')
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--output_dir', default='outputs/figures')
    args = parser.parse_args()
    style()
    out = Path(args.output_dir)

    csv_path = Path(args.explanations_csv)
    if csv_path.exists():
        rows = read_csv(csv_path)
        json_path = Path(args.explanations_json) if args.explanations_json else csv_path.with_suffix('.json')
        with open(json_path, encoding='utf-8') as handle:
            details = {d['sample_id']: d for d in json.load(handle)}
        by_id = {r['sample_id']: r for r in rows}
        samples = args.samples or pick_samples(rows)
        print(f'explanation cards for samples: {samples}')
        for sample_id in samples:
            explanation_card(by_id[sample_id], details[sample_id], args.top_k,
                             out / f'p3_card_{sample_id}.png')
        contribution_shares(rows, out / 'p3_modality_contributions.png')
        position_profile(list(details.values()), out / 'p3_evidence_position_profile.png')
    if Path(args.faithfulness_json).exists():
        with open(args.faithfulness_json, encoding='utf-8') as handle:
            faithfulness_chart(json.load(handle), out / 'p3_faithfulness.png')
    if Path(args.missing_csv).exists():
        rows = read_csv(args.missing_csv)
        missing_curves(rows, out / 'p2_missing_ratio.png')
        missing_positions(rows, out / 'p2_missing_position.png')


if __name__ == '__main__':
    main()
