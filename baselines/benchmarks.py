"""
Benchmarks (Step 7 of the replication guide):

  FM   Full Modality       - no missing modalities. lambda_missing = 0
  PM   Partial Modality    - train only on available modalities, no
                             compensation.
  ZF   Zero Filling        - fill missing modality input with zeros.
  PMM  Proposed method     - full prototype-substitution algorithm
                             (see fl/pmm.py, train.py::mmo_fl_train)

NOTE ON PM vs ZF (implementation assumption):
The doc's Step 7 table says PM should "skip" the missing encoder's output
rather than zero-fill it. Concatenating a shorter feature vector would
change the Head's input dimension between missing/non-missing rounds,
which nn.Linear can't handle directly. The most faithful reading that
keeps a fixed-size Head is:

  - ZF: missing modality's *raw input* is zeroed, then run through its
    encoder as normal — the encoder still produces a (learned, non-zero)
    feature vector and receives gradients.
  - PM: the missing modality's encoder is *not called at all* (no forward,
    no gradient); its slot in the concatenated feature vector is padded
    with a fixed zero vector, so only the available-modality encoder(s)
    and the Head receive gradient updates that round.

If you have access to the original paper's exact equations for PM, adjust
`run_pm` below accordingly — this is a reasonable but not verbatim
reconstruction.
"""

import copy
import torch
import torch.nn as nn

from fl.server import fedavg


def _pm_forward_and_backward(model, batch, missing, E, lr):
    """PM: missing modality encoder is skipped entirely; its feature slot
    is zero-padded before the Head. No gradients reach that encoder."""
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    modality_data, labels = batch

    for _ in range(E):
        optimizer.zero_grad()
        features = []
        for i, (enc, x) in enumerate(zip(model.encoders, modality_data)):
            if i == missing:
                with torch.no_grad():
                    feat_dim = enc.fc.out_features
                features.append(torch.zeros(x.shape[0], feat_dim, device=x.device))
            else:
                features.append(enc(x))
        z = torch.cat(features, dim=-1)
        logits = model.head(z)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()


def run_fm(clients, server_model, T, E, lr, K=5, verbose=True,
           test_batch=None, decay=0.95, lr_min=0.001):
    """Full Modality baseline: no missing-modality rounds at all."""
    from fl.client import do_local_update

    history = []
    for t in range(T):
        lr_t = max(lr * (decay ** t), lr_min)
        global_state = copy.deepcopy(server_model.state_dict())
        local_states = []
        for client in clients:
            client.model.load_state_dict(global_state)
            client.buffer.update(n_new=20)
            batch = client.buffer.get_all()
            do_local_update(client.model, batch, E, lr_t, missing=None)
            local_states.append(copy.deepcopy(client.model.state_dict()))
        server_model = fedavg(server_model, local_states)
        entry = {'round': t + 1}
        if test_batch is not None:
            from train import evaluate
            entry['acc'], preds = evaluate(server_model, test_batch)
            counts = {}
            for c in preds.tolist(): counts[c] = counts.get(c, 0) + 1
            entry['pred_dist'] = dict(sorted(counts.items()))

        history.append(entry)
        if verbose and (t % max(1, T // 10) == 0):
            acc_str = f" acc={entry['acc']:.4f} n_cls={len(entry['pred_dist'])}/6" if test_batch else ""
            print(f"  [FM] round {t+1}/{T}" + acc_str)
    return server_model, history


def run_pm(clients, server_model, T, E, lr, lambda_missing, K=5, verbose=True,
           test_batch=None, n_modalities=2, decay=0.95, lr_min=0.001):
    """Partial Modality baseline: missing modality skipped, no compensation."""
    history = []
    for t in range(T):
        lr_t = max(lr * (decay ** t), lr_min)
        global_state = copy.deepcopy(server_model.state_dict())
        local_states = []
        round_missing = (int(torch.randint(0, n_modalities, (1,)).item())
                         if torch.rand(1).item() < lambda_missing else None)
        for client in clients:
            client.model.load_state_dict(global_state)
            client.buffer.update(n_new=20)
            batch = client.buffer.get_all()
            # Round-level: one modality missing for ALL clients, or none (paper Sec VII-B)
            if round_missing is None:
                from fl.client import do_local_update
                do_local_update(client.model, batch, E, lr, missing=None)
            else:
                _pm_forward_and_backward(client.model, batch, round_missing, E, lr_t)
            local_states.append(copy.deepcopy(client.model.state_dict()))
        server_model = fedavg(server_model, local_states)
        entry = {'round': t + 1}
        if test_batch is not None:
            from train import evaluate
            entry['acc'], _ = evaluate(server_model, test_batch)
        history.append(entry)
        if verbose and (t % max(1, T // 10) == 0):
            print(f"  [PM] round {t+1}/{T}" + (f"  acc={entry.get('acc', 0):.4f}" if test_batch else ""))
    return server_model, history


def run_zf(clients, server_model, T, E, lr, lambda_missing, K=5, verbose=True,
           test_batch=None, n_modalities=2, decay=0.95, lr_min=0.001):
    """Zero Filling baseline: missing modality's raw input replaced with zeros."""
    history = []
    for t in range(T):
        lr_t = max(lr * (decay ** t), lr_min)
        global_state = copy.deepcopy(server_model.state_dict())
        local_states = []
        round_missing = (int(torch.randint(0, n_modalities, (1,)).item())
                         if torch.rand(1).item() < lambda_missing else None)
        for client in clients:
            client.model.load_state_dict(global_state)
            client.buffer.update(n_new=20)
            modality_data, labels = client.buffer.get_all()
            if round_missing is not None:
                modality_data = [
                    x if i != round_missing else torch.zeros_like(x)
                    for i, x in enumerate(modality_data)
                ]
            optimizer = torch.optim.SGD(client.model.parameters(), lr=lr_t)
            criterion = nn.CrossEntropyLoss()
            N = labels.shape[0]
            batch_size = 64
            for _ in range(E):
                perm = torch.randperm(N)
                for start in range(0, N, batch_size):
                    idx = perm[start:start + batch_size]
                    mb_data = [x[idx] for x in modality_data]
                    mb_labels = labels[idx]
                    optimizer.zero_grad()
                    logits = client.model(mb_data)
                    loss = criterion(logits, mb_labels)
                    loss.backward()
                    optimizer.step()
            local_states.append(copy.deepcopy(client.model.state_dict()))
        server_model = fedavg(server_model, local_states)
        entry = {'round': t + 1}
        if test_batch is not None:
            from train import evaluate
            entry['acc'], _ = evaluate(server_model, test_batch)
        history.append(entry)
        if verbose and (t % max(1, T // 10) == 0):
            print(f"  [ZF] round {t+1}/{T}" + (f"  acc={entry.get('acc', 0):.4f}" if test_batch else ""))
    return server_model, history


def run_pmm(clients, server_model, T, E, lr, lambda_missing, K=5, verbose=True,
            test_batch=None, n_modalities=2, decay=0.95, lr_min=0.001):
    """PMM: delegates to the full algorithm in train.mmo_fl_train."""
    from train import mmo_fl_train
    server_model, prototypes, history = mmo_fl_train(
        clients, server_model, T, E, lr, lambda_missing, K=K, verbose=verbose,
        test_batch=test_batch, n_modalities=n_modalities,
        decay=decay, lr_min=lr_min,
    )
    return server_model, prototypes, history
