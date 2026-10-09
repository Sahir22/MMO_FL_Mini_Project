"""
Step 9 — Ablation experiments (paper Section VII-E).

Each function runs one ablation study, sweeping the stated parameter and
recording accuracy per round for each value, then saves a CSV. Plotting
(Fig. 5-8 style) is left as a `plot_results()` you can point at any CSV.

Run all four (small/fast settings by default — bump --rounds for the
paper's real T=100):

    python ablation/run_ablations.py --study missing_rate --rounds 20
    python ablation/run_ablations.py --study noniid --rounds 20
    python ablation/run_ablations.py --study quant --rounds 20
    python ablation/run_ablations.py --study delay --rounds 20
"""

import argparse
import copy
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from train import mmo_fl_train, make_synthetic_clients, make_ucihar_clients, evaluate
from models.ucihar_model import AccelEncoder, GyroEncoder, HeadEncoder, MMOFLModel


def _fresh_server(num_classes=6):
    return MMOFLModel([AccelEncoder(), GyroEncoder()], HeadEncoder(feat_dim=256, num_classes=num_classes))


def _get_clients(args, seed):
    if args.dataset == "synthetic":
        return make_synthetic_clients(args.clients, seed=seed), None, 6
    return make_ucihar_clients(args.data_path, K=args.clients, alpha=args.alpha,
                                initial_per_client=args.initial_per_client,
                                buffer_size=args.buffer_size, seed=seed)


def ablation_missing_rate(args):
    """1. Vary lambda in {0.3, 0.5, 0.7} (Fig. 5)."""
    rows = []
    for lam in [0.3, 0.5, 0.7]:
        for seed in range(args.seeds):
            clients, test_batch, num_classes = _get_clients(args, seed=seed)
            server = _fresh_server(num_classes)
            _, _, hist = mmo_fl_train(clients, server, T=args.rounds, E=1, lr=0.1,
                                      lambda_missing=lam, K=args.clients, verbose=False,
                                      test_batch=test_batch)
            for entry in hist:
                rows.append({"lambda": lam, "seed": seed,
                              "round": entry['round'], "acc": entry.get('acc', '')})
            final_acc = hist[-1].get('acc')
            print(f"lambda={lam} seed={seed}: final_acc={final_acc}")
    _save_csv(rows, "ablation_missing_rate.csv")


def ablation_noniid(args):
    """2. Vary alpha in {1, 5, 10} (Fig. 6)."""
    rows = []
    for alpha in [1, 5, 10]:
        args.alpha = alpha
        for seed in range(args.seeds):
            clients, test_batch, num_classes = _get_clients(args, seed=seed)
            server = _fresh_server(num_classes)
            _, _, hist = mmo_fl_train(clients, server, T=args.rounds, E=1, lr=0.1,
                                      lambda_missing=0.5, K=args.clients, verbose=False,
                                      test_batch=test_batch)
            for entry in hist:
                rows.append({"alpha": alpha, "seed": seed,
                              "round": entry['round'], "acc": entry.get('acc', '')})
            final_acc = hist[-1].get('acc')
            print(f"alpha={alpha} seed={seed}: final_acc={final_acc}")
    _save_csv(rows, "ablation_noniid.csv")


def _quantize(tensor, bits):
    """Uniform scalar quantization to `bits` bits, per the doc's Step 9.3."""
    qmin, qmax = tensor.min(), tensor.max()
    levels = 2 ** bits
    if qmax == qmin:
        return tensor.clone()
    scale = (qmax - qmin) / (levels - 1)
    q = torch.round((tensor - qmin) / scale)
    return q * scale + qmin


def ablation_quantized_upload(args):
    """3. Quantize prototype vectors to b in {2, 4} bits before server
    aggregation (Fig. 7). This patches fl.pmm.compute_local_prototypes'
    output before it's sent to the server-side aggregation step."""
    import fl.pmm as pmm_module

    rows = []
    for bits in [2, 4]:
        for seed in range(args.seeds):
            clients, test_batch, num_classes = _get_clients(args, seed=seed)
            server = _fresh_server(num_classes)

            original_compute = pmm_module.compute_local_prototypes

            def quantized_compute(model, batch, _bits=bits):
                protos = original_compute(model, batch)
                return {k: _quantize(v, _bits) for k, v in protos.items()}

            pmm_module.compute_local_prototypes = quantized_compute
            try:
                _, _, hist = mmo_fl_train(clients, server, T=args.rounds, E=1, lr=0.1,
                                          lambda_missing=0.5, K=args.clients, verbose=False,
                                          test_batch=test_batch)
            finally:
                pmm_module.compute_local_prototypes = original_compute

            for entry in hist:
                rows.append({"bits": bits, "seed": seed,
                              "round": entry['round'], "acc": entry.get('acc', '')})
            final_acc = hist[-1].get('acc')
            print(f"bits={bits} seed={seed}: final_acc={final_acc}")
    _save_csv(rows, "ablation_quant.csv")


def ablation_delayed_update(args):
    """4. OPC runs every DL in {2, 4} rounds instead of every round (Fig. 8)."""
    rows = []
    for DL in [2, 4]:
        for seed in range(args.seeds):
            clients, test_batch, num_classes = _get_clients(args, seed=seed)
            server = _fresh_server(num_classes)
            prototypes = {}
            prototype_counts = {}
            hist = []

            from fl.client import do_local_update
            from fl.server import fedavg, update_global_prototypes
            from fl.pmm import compute_local_prototypes, get_proto_features

            for t in range(args.rounds):
                global_state = copy.deepcopy(server.state_dict())
                run_opc_this_round = (t % DL == 0)

                local_states, new_protos = [], {}
                for client in clients:
                    client.model.load_state_dict(global_state)
                    client.buffer.update(n_new=20)
                    batch = client.buffer.get_all()
                    _, labels = batch

                    missing = (int(torch.randint(0, 2, (1,)).item())
                               if torch.rand(1).item() < 0.5 else None)

                    if missing is None:
                        if run_opc_this_round:
                            local_protos = compute_local_prototypes(client.model, batch)
                            for key, val in local_protos.items():
                                new_protos.setdefault(key, []).append(val)
                        do_local_update(client.model, batch, 1, 0.1, missing=None)
                    else:
                        proto_feat = get_proto_features(labels, missing, prototypes)
                        do_local_update(client.model, batch, 1, 0.1,
                                        missing=missing, proto_features=proto_feat)

                    local_states.append(copy.deepcopy(client.model.state_dict()))

                server = fedavg(server, local_states)
                if run_opc_this_round:
                    prototypes, prototype_counts = update_global_prototypes(
                        prototypes, prototype_counts, new_protos
                    )

                acc = evaluate(server, test_batch) if test_batch else None
                hist.append({'round': t + 1, 'acc': acc})

            for entry in hist:
                rows.append({"DL": DL, "seed": seed,
                              "round": entry['round'], "acc": entry.get('acc', '')})
            print(f"DL={DL} seed={seed}: final_acc={hist[-1].get('acc')}")
    _save_csv(rows, "ablation_delay.csv")


def _save_csv(results, filename):
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"saved {path}")


STUDIES = {
    "missing_rate": ablation_missing_rate,
    "noniid": ablation_noniid,
    "quant": ablation_quantized_upload,
    "delay": ablation_delayed_update,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--study", choices=list(STUDIES.keys()) + ["all"], default="all")
    p.add_argument("--dataset", choices=["synthetic", "ucihar"], default="synthetic")
    p.add_argument("--data-path", default="./UCI HAR Dataset")
    p.add_argument("--clients", type=int, default=5)
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--initial-per-client", type=int, default=500)
    p.add_argument("--buffer-size", type=int, default=500)
    p.add_argument("--seeds", type=int, default=1,
                   help="Seeds to average over (paper uses 10)")
    args = p.parse_args()

    studies = STUDIES.keys() if args.study == "all" else [args.study]
    for name in studies:
        print(f"\n=== ablation: {name} ===")
        STUDIES[name](args)


if __name__ == "__main__":
    main()
