"""
plot.py — Generate accuracy-vs-rounds figures from CSVs produced by
compare.py and ablation/run_ablations.py.

Usage
-----
# Main comparison curves (Figs 3-4 style: FM vs PM vs ZF vs PMM)
python plot.py --csv results/compare_ucihar_T100_lam0.5_alpha1.0.csv \
               --out results/fig_compare.png

# Ablation curves (Figs 5-8)
python plot.py --csv ablation/results/ablation_missing_rate.csv \
               --hue lambda --out results/fig5_missing_rate.png
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
LINE_STYLES = ['-', '--', '-.', ':']


def plot_accuracy_curves(csv_path, out_path, hue_col=None, title=None):
    df = pd.read_csv(csv_path)

    if 'acc' not in df.columns or df['acc'].isna().all():
        print("No accuracy data in CSV (synthetic run?) — nothing to plot.")
        return

    df = df[df['acc'].notna()].copy()
    df['acc'] = df['acc'].astype(float)

    # Determine grouping column
    if hue_col and hue_col in df.columns:
        group_col = hue_col
    elif 'method' in df.columns:
        group_col = 'method'
    else:
        # Fall back to first non-round, non-seed, non-acc column
        candidates = [c for c in df.columns if c not in ('round', 'seed', 'acc')]
        group_col = candidates[0] if candidates else None

    fig, ax = plt.subplots(figsize=(7, 4.5))

    groups = sorted(df[group_col].unique()) if group_col else ['all']
    for i, grp in enumerate(groups):
        subset = df[df[group_col] == grp] if group_col else df
        # Average over seeds if present
        pivot = subset.groupby('round')['acc'].agg(['mean', 'std']).reset_index()
        label = f"{group_col}={grp}" if group_col else "accuracy"
        color = COLORS[i % len(COLORS)]
        ls = LINE_STYLES[i % len(LINE_STYLES)]
        ax.plot(pivot['round'], pivot['mean'], label=label, color=color, ls=ls, lw=1.8)
        if 'std' in pivot and pivot['std'].notna().any():
            ax.fill_between(pivot['round'],
                            pivot['mean'] - pivot['std'],
                            pivot['mean'] + pivot['std'],
                            alpha=0.12, color=color)

    ax.set_xlabel("Communication round")
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    ax.set_title(title or os.path.basename(csv_path))
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Path to input CSV")
    p.add_argument("--out", required=True, help="Output PNG path")
    p.add_argument("--hue", default=None,
                   help="Column to use for line grouping (default: 'method')")
    p.add_argument("--title", default=None)
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plot_accuracy_curves(args.csv, args.out, hue_col=args.hue, title=args.title)


if __name__ == "__main__":
    main()
