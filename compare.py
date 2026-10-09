"""
compare.py — Run all four baselines (FM, PM, ZF, PMM) on the same data split
and output a comparison table + per-round CSV (replicates Tables II/III and
the main accuracy-vs-rounds curves from the paper).

Usage
-----
Smoke test (no dataset needed):

    python compare.py --dataset synthetic --rounds 10

UCI-HAR (download "UCI HAR Dataset.zip" first):

    python compare.py --dataset ucihar --data-path "./UCI HAR Dataset" \
        --rounds 100 --clients 5 --alpha 1 --lambda-missing 0.5

MVSA-Single:

    python compare.py --dataset mvsa --data-path ./MVSA_Single \
        --rounds 120 --clients 5 --alpha 1 --lambda-missing 0.5

Multi-seed (paper uses 10):

    python compare.py --dataset ucihar --rounds 100 --seeds 10

Output
------
  results/compare_<dataset>_T<rounds>_lam<lambda>_alpha<alpha>.csv
  results/compare_<dataset>_T<rounds>_lam<lambda>_alpha<alpha>_summary.txt
"""

import argparse
import copy
import csv
import os
import statistics
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import (make_synthetic_clients, make_ucihar_clients,
                   make_mvsa_clients, evaluate)
from models.ucihar_model import AccelEncoder, GyroEncoder, HeadEncoder, MMOFLModel
from baselines.benchmarks import run_fm, run_pm, run_zf, run_pmm


def _fresh_server_ucihar(num_classes):
    return MMOFLModel(
        [AccelEncoder(), GyroEncoder()],
        HeadEncoder(feat_dim=256, num_classes=num_classes)
    )


def _get_clients(args, seed):
    if args.dataset == "synthetic":
        clients = make_synthetic_clients(args.clients, seed=seed)
        return clients, None, 6
    if args.dataset == "ucihar":
        return make_ucihar_clients(
            args.data_path, K=args.clients, alpha=args.alpha,
            initial_per_client=args.initial_per_client,
            buffer_size=args.buffer_size, seed=seed,
        )
    if args.dataset == "mvsa":
        return make_mvsa_clients(
            args.data_path, K=args.clients, alpha=args.alpha,
            initial_per_client=args.initial_per_client,
            buffer_size=args.buffer_size, seed=seed,
        )
    raise ValueError(f"Unknown dataset: {args.dataset}")


def _fresh_server(args, num_classes):
    if args.dataset in ("synthetic", "ucihar"):
        return _fresh_server_ucihar(num_classes)
    from models.mvsa_model import build_mvsa_model
    return build_mvsa_model()


def run_comparison(args, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    clients, test_batch, num_classes = _get_clients(args, seed)

    results = {}  # method -> list of per-round dicts

    os.makedirs("results/models", exist_ok=True)

    # --- FM ---
    print(f"  [FM] seed={seed}")
    fm_clients, fm_test, _ = _get_clients(args, seed)
    fm_server = _fresh_server(args, num_classes)
    fm_server, fm_hist = run_fm(
        fm_clients, fm_server, T=args.rounds, E=args.E, lr=args.lr,
        verbose=False, test_batch=fm_test,
        decay=args.decay, lr_min=args.lr_min
    )
    results['FM'] = fm_hist
    torch.save(fm_server.state_dict(), f"results/models/fm_seed{seed}.pt")

    # --- PM ---
    print(f"  [PM] seed={seed}")
    pm_clients, pm_test, _ = _get_clients(args, seed)
    pm_server = _fresh_server(args, num_classes)
    pm_server, pm_hist = run_pm(
        pm_clients, pm_server, T=args.rounds, E=args.E, lr=args.lr,
        lambda_missing=args.lambda_missing, verbose=False, test_batch=pm_test,
        decay=args.decay, lr_min=args.lr_min
    )
    results['PM'] = pm_hist
    torch.save(pm_server.state_dict(), f"results/models/pm_seed{seed}.pt")

    # --- ZF ---
    print(f"  [ZF] seed={seed}")
    zf_clients, zf_test, _ = _get_clients(args, seed)
    zf_server = _fresh_server(args, num_classes)
    zf_server, zf_hist = run_zf(
        zf_clients, zf_server, T=args.rounds, E=args.E, lr=args.lr,
        lambda_missing=args.lambda_missing, verbose=False, test_batch=zf_test,
        decay=args.decay, lr_min=args.lr_min
    )
    results['ZF'] = zf_hist
    torch.save(zf_server.state_dict(), f"results/models/zf_seed{seed}.pt")    

    # --- PMM ---
    print(f"  [PMM] seed={seed}")
    pmm_clients, pmm_test, _ = _get_clients(args, seed)
    pmm_server = _fresh_server(args, num_classes)
    pmm_server, _, pmm_hist = run_pmm(
        pmm_clients, pmm_server, T=args.rounds, E=args.E, lr=args.lr,
        lambda_missing=args.lambda_missing, verbose=False, test_batch=pmm_test,
        decay=args.decay, lr_min=args.lr_min
    )
    results['PMM'] = pmm_hist
    torch.save(pmm_server.state_dict(), f"results/models/pmm_seed{seed}.pt")

    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["synthetic", "ucihar", "mvsa"], default="synthetic")
    p.add_argument("--data-path", default="./UCI HAR Dataset")
    p.add_argument("--clients", type=int, default=5)
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--E", type=int, default=1)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--decay", type=float, default=0.95)
    p.add_argument("--lr-min", type=float, default=0.001)
    p.add_argument("--alpha", type=float, default=10.0)
    p.add_argument("--lambda-missing", type=float, default=0.5)
    p.add_argument("--initial-per-client", type=int, default=500)
    p.add_argument("--buffer-size", type=int, default=2000)
    p.add_argument("--seeds", type=int, default=1,
                   help="Seeds to average over (paper uses 10)")
    args = p.parse_args()

    methods = ['FM', 'PM', 'ZF', 'PMM']
    # seed -> method -> list of per-round dicts
    all_results = {}

    for s in range(args.seeds):
        print(f"=== seed {s} ===")
        all_results[s] = run_comparison(args, seed=s)

    # --- Build per-round CSV (one row per round per method per seed) ---
    os.makedirs("results", exist_ok=True)
    tag = f"{args.dataset}_T{args.rounds}_lam{args.lambda_missing}_alpha{args.alpha}"
    csv_path = os.path.join("results", f"compare_{tag}.csv")

    rows = []
    for s, method_hists in all_results.items():
        for method, hist in method_hists.items():
            for entry in hist:
                rows.append({
                    'seed': s,
                    'method': method,
                    'round': entry['round'],
                    'acc': entry.get('acc', ''),
                })

    if rows:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['seed', 'method', 'round', 'acc'])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nPer-round CSV saved: {csv_path}")

    # --- Summary table (final-round accuracy, mean ± std over seeds) ---
    summary_path = os.path.join("results", f"compare_{tag}_summary.txt")
    lines = [f"Dataset: {args.dataset}  T={args.rounds}  λ={args.lambda_missing}  α={args.alpha}  seeds={args.seeds}\n",
             f"{'Method':<8}  {'Mean Acc':>10}  {'Std':>8}"]
    for method in methods:
        final_accs = [
            all_results[s][method][-1].get('acc')
            for s in range(args.seeds)
            if all_results[s][method][-1].get('acc') is not None
        ]
        if final_accs:
            mean = statistics.mean(final_accs)
            std = statistics.pstdev(final_accs) if len(final_accs) > 1 else 0.0
            lines.append(f"{method:<8}  {mean:>10.4f}  {std:>8.4f}")
        else:
            lines.append(f"{method:<8}  {'N/A':>10}  {'N/A':>8}")

    summary = '\n'.join(lines)
    print('\n' + summary)
    with open(summary_path, 'w') as f:
        f.write(summary + '\n')
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
