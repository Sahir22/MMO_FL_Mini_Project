"""
train.py - Main experiment runner for MMO-FL + PMM replication.

Paper: "Multimodal Online Federated Learning With Modality Missing in IoT"
       Wang et al., IEEE TMC 2026

Usage
-----
Smoke test (no dataset download needed, uses random synthetic data):

    python train.py --dataset synthetic --rounds 5 --clients 3

UCI-HAR run (after downloading & extracting "UCI HAR Dataset.zip"):

    python train.py --dataset ucihar --rounds 100 --clients 5 \
        --alpha 1 --lambda-missing 0.5 --E 1

MVSA-Single run (after downloading MVSA-Single dataset):

    python train.py --dataset mvsa --data-path ./MVSA_Single \
        --rounds 120 --clients 5 --alpha 1 --lambda-missing 0.5 --E 1

10-seed evaluation harness (paper reports mean over 10 seeds):

    python train.py --dataset ucihar --rounds 100 --seeds 10
"""

import argparse
import copy
import statistics

import torch
import numpy as np

from fl.client import Client, StreamingBuffer, DataPool, do_local_update
from fl.server import fedavg, update_global_prototypes
from fl.pmm import compute_local_prototypes, get_proto_features
from models.ucihar_model import AccelEncoder, GyroEncoder, HeadEncoder, MMOFLModel


# ---------------------------------------------------------------------------
# MMO-FL training loop (paper Section IV / Algorithm 1)
# ---------------------------------------------------------------------------

def mmo_fl_train(clients, server_model, T, E, lr, lambda_missing, K=5,
                  decay=0.95, lr_min=0.001, verbose=True,
                  test_batch=None, n_modalities=2, use_ops=True):
    """
    Returns (server_model, prototypes, history) where history is a list of
    dicts with keys 'round', 'lr', 'n_missing_clients', and optionally 'acc'
    (present when test_batch is provided).
    """
    prototypes = {}  # persistent global prototypes: {(m, c): tensor(128,)}
    prototype_counts = {}  # per-prototype update counts for running average
    history = []
    num_classes = server_model.head.net[-1].out_features

    for t in range(T):
        lr_t = max(lr * (decay ** t), lr_min)

        # 1. Server broadcasts current model
        global_state = copy.deepcopy(server_model.state_dict())

        # 2. Correlated missing modality
        if torch.rand(1).item() < lambda_missing:
            missing_m = int(torch.randint(0, n_modalities, (1,)).item())
            client_missing_list = [missing_m] * len(clients)
        else:        
            client_missing_list = [None] * len(clients)

        local_states = []
        new_local_prototypes = {}  # for OPC
        n_missing = sum(m is not None for m in client_missing_list)
        client_losses = []

        for client, client_missing in zip(clients, client_missing_list):
            client.model.load_state_dict(global_state)
            client.model.train()

            client.buffer.update(n_new=20)
            batch = client.buffer.get_all()
            _, labels = batch

            if client_missing is None:
                # Full modality client: run OPC and standard OGD
                local_protos = compute_local_prototypes(client.model, batch)
                for key, val in local_protos.items():
                    new_local_prototypes.setdefault(key, []).append(val)
                loss_val = do_local_update(client.model, batch, E, lr_t, missing=None)
            else:
                # Missing modality client: use OPS only if prototypes exist for
                # this modality; otherwise treat as full modality (cold-start guard)
                # Injecting zero/random features in early rounds corrupts training
                has_protos = use_ops and any((client_missing, c) in prototypes for c in range(num_classes))
                if has_protos:
                    proto_feat = get_proto_features(
                        labels, client_missing, prototypes, feat_dim=128
                    )
                    loss_val = do_local_update(
                        client.model, batch, E, lr_t,
                        missing=client_missing, proto_features=proto_feat
                    )
                else:
                    
                    if use_ops:
                    #Cold start guard, prototypes not yet available, ZF rather than full modality
                        loss_val = do_local_update(client.model, batch, E, lr_t, missing=client_missing)

                    else:
                        #ZF_baseline: pass missing index so do_local_update zeroes the raw input
                        loss_val = do_local_update(client.model, batch, E, lr_t, missing=client_missing)

            client_losses.append(loss_val)
            local_states.append(copy.deepcopy(client.model.state_dict()))

        # 3. Server aggregates (FedAvg, eq. 7)
        server_model = fedavg(server_model, local_states)
        nan_weights = any(torch.isnan(v).any().item() 
                          for v in server_model.state_dict().values())

        # 4. Update persistent prototypes (OPC server side, eq. 16-17)
        prototypes, prototype_counts = update_global_prototypes(
           prototypes, prototype_counts, new_local_prototypes
       )


       # 5. Record per-round metrics
        valid_losses = [l for l in client_losses if l is not None]
        mean_loss = sum(valid_losses) / len(valid_losses) if valid_losses else float('nan')
        entry = {'round': t + 1, 'lr': lr_t, 'n_missing_clients': n_missing,
                'mean_loss': mean_loss, 'nan_weights': nan_weights}
        if test_batch is not None:
           entry['acc'], preds = evaluate(server_model, test_batch)
           counts = {}
           for c in preds.tolist():
               counts[c] = counts.get(c, 0) + 1
           entry['pred_dist'] = dict(sorted(counts.items()))


        history.append(entry)


        if verbose and (t % max(1, T // 10) == 0 or t == T - 1):
            acc_str = ""
            if 'acc' in entry:
                n_cls = len(entry['pred_dist'])
                acc_str = f"  acc={entry['acc']:.4f}  n_cls={n_cls}/6  dist={entry['pred_dist']}"
            nan_str = "  *** NaN WEIGHTS ***" if nan_weights else ""
            print(f"  round {t+1}/{T}  lr={lr_t:.4f}  loss={mean_loss:.4f}  missing_clients={n_missing}/{len(clients)}{acc_str}{nan_str}")


    return server_model, prototypes, history


def evaluate(model, test_batch, prototypes=None, missing_m=None):
    model.eval()
    modality_data, labels = test_batch
    with torch.no_grad():
        if missing_m is not None and prototypes:
            # OPS at inference: inject prototype features for missing modality
            proto_feat = get_proto_features(labels, missing_m, prototypes, feat_dim=128)
            logits = model.forward_ops(modality_data, missing_m, proto_feat)
        else:
            logits = model(modality_data)
        preds = logits.argmax(1)
        correct = (logits.argmax(1) == labels).sum().item()
        total = len(labels)
    return correct / total, preds


# ---------------------------------------------------------------------------
# Data construction helpers
# ---------------------------------------------------------------------------

def make_synthetic_clients(K, num_classes=6, samples_per_client=300,
                            buffer_size=100, seed=0):
    """Random data with UCI-HAR shapes, for smoke-testing the pipeline."""
    rng = np.random.RandomState(seed)
    clients = []
    for k in range(K):
        init_data = [
            ([rng.randn(128, 3).astype(np.float32), rng.randn(128, 3).astype(np.float32)],
             int(rng.randint(0, num_classes)))
            for _ in range(buffer_size)
        ]
        long_term = [
            ([rng.randn(128, 3).astype(np.float32), rng.randn(128, 3).astype(np.float32)],
             int(rng.randint(0, num_classes)))
            for _ in range(samples_per_client)
        ]
        pool = DataPool(long_term, seed=seed + k)
        buffer = StreamingBuffer(init_data, max_size=buffer_size, long_term_source=pool)
        model = MMOFLModel(
            [AccelEncoder(), GyroEncoder()],
            HeadEncoder(feat_dim=256, num_classes=num_classes)
        )
        clients.append(Client(k, model, buffer))
    return clients


def make_ucihar_clients(data_path, K=5, alpha=1.0, initial_per_client=2000,
                         buffer_size=500, seed=0):
    from data.ucihar import load_ucihar, build_client_splits

    data = load_ucihar(data_path)
    accel, gyro, labels = data['train']
    splits = build_client_splits(accel, gyro, labels, K=K, alpha=alpha,
                                  initial_per_client=initial_per_client, seed=seed)

    num_classes = int(labels.max()) + 1
    clients = []
    for k, split in enumerate(splits):
        pool = DataPool(split['long_term'], seed=seed + k)
        buffer = StreamingBuffer(split['initial'], max_size=buffer_size, long_term_source=pool)
        model = MMOFLModel(
            [AccelEncoder(), GyroEncoder()],
            HeadEncoder(feat_dim=256, num_classes=num_classes)
        )
        clients.append(Client(k, model, buffer))

    test_accel, test_gyro, test_labels = data['test']
    test_batch = (
        [torch.tensor(test_accel, dtype=torch.float32),
         torch.tensor(test_gyro, dtype=torch.float32)],
        torch.tensor(test_labels, dtype=torch.long),
    )
    return clients, test_batch, num_classes


def make_mvsa_clients(data_path, K=5, alpha=5.0, initial_per_client=1500,
                       buffer_size=800, seed=0):
    from data.mvsa import load_mvsa, build_tokenizer, build_client_splits
    from models.mvsa_model import ImageEncoder, TextEncoder, build_mvsa_model

    images, texts, labels = load_mvsa(data_path, image_size=64)
    vocab, encode_fn = build_tokenizer(texts, vocab_size=10000)

    # Convert images to (N, 3, H, W) float and tokenize texts
    imgs_chw = torch.tensor(
        images.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
    )
    text_ids = torch.tensor(
        np.array([encode_fn(t) for t in texts], dtype=np.int64)
    )

    n = len(labels)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    test_idx  = perm[:n // 5]
    train_idx = perm[n // 5:]


    splits = build_client_splits(
        imgs_chw[train_idx], text_ids[train_idx], labels[train_idx],
        K=K, alpha=alpha, initial_per_client=initial_per_client, seed=seed,
    )


    num_classes = 3
    clients = []
    for k, split in enumerate(splits):
        pool = DataPool(split['long_term'], seed=seed + k)
        buffer = StreamingBuffer(split['initial'], max_size=buffer_size,
                                  long_term_source=pool)
        model = build_mvsa_model(vocab_size=len(vocab))
        clients.append(Client(k, model, buffer))


    test_batch = (
        [imgs_chw[test_idx], text_ids[test_idx]],
        torch.tensor(labels[test_idx], dtype=torch.long),
    )
    return clients, test_batch, num_classes



# ---------------------------------------------------------------------------
# Single run / multi-seed harness
# ---------------------------------------------------------------------------

def run_once(args, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    if args.dataset == "synthetic":
        clients = make_synthetic_clients(args.clients, seed=seed)
        num_classes = 6
        test_batch = None
        server_model = MMOFLModel(
            [AccelEncoder(), GyroEncoder()],
            HeadEncoder(feat_dim=256, num_classes=num_classes)
        )
    elif args.dataset == "ucihar":
        clients, test_batch, num_classes = make_ucihar_clients(
            args.data_path, K=args.clients, alpha=args.alpha,
            initial_per_client=args.initial_per_client,
            buffer_size=args.buffer_size, seed=seed,
        )
        server_model = MMOFLModel(
            [AccelEncoder(), GyroEncoder()],
            HeadEncoder(feat_dim=256, num_classes=num_classes)
        )
    elif args.dataset == "mvsa":
        clients, test_batch, num_classes = make_mvsa_clients(
            args.data_path, K=args.clients, alpha=args.alpha,
            initial_per_client=args.initial_per_client,
            buffer_size=args.buffer_size, seed=seed,
        )
        from models.mvsa_model import build_mvsa_model
        server_model = build_mvsa_model()
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    n_modalities = 2  # both datasets use exactly 2 modalities
    server_model, prototypes, history = mmo_fl_train(
        clients, server_model, T=args.rounds, E=args.E, lr=args.lr,
        lambda_missing=args.lambda_missing, K=args.clients,
        decay=args.decay, lr_min=args.lr_min, verbose=args.verbose,
        test_batch=test_batch, n_modalities=n_modalities, 
        use_ops=not args.no_ops,
    )

    if getattr(args, 'save_model', None):
        torch.save({
            'model_state': server_model.state_dict(),
            'prototypes': prototypes,
            'config': vars(args),
        }, args.save_model)
        print(f"  saved model to {args.save_model}")

    if test_batch is not None:
        acc = history[-1]['acc']
        return acc, history
    return None, history


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
    p.add_argument("--initial-per-client", type=int, default=2000)
    p.add_argument("--buffer-size", type=int, default=500)
    p.add_argument("--seeds", type=int, default=1,
                   help="Number of seeds to average over (paper uses 10)")
    p.add_argument("--verbose", action="store_true", default=True)
    p.add_argument("--no-ops", action="store_true", default=False, help="Disable OPS (use zero-filling for missing modalities instead)")
    p.add_argument("--save-model", default=None, help="Path to save the trained Model (eg. model.pt)")
    args = p.parse_args()

    accs = []
    for s in range(args.seeds):
        print(f"=== seed {s} ===")
        acc, history = run_once(args, seed=s)
        if acc is not None:
            print(f"  test accuracy: {acc:.4f}")
            accs.append(acc)
        else:
            print("  (synthetic run, no held-out accuracy — pipeline check only)")

    if accs:
        print(f"\nMean accuracy over {len(accs)} seed(s): "
              f"{statistics.mean(accs):.4f} "
              f"(+/- {statistics.pstdev(accs):.4f})")


if __name__ == "__main__":
    main()
