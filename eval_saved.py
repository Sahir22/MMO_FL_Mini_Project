import torch, sys, os
sys.path.insert(0, ".")
from train import make_ucihar_clients, evaluate
from models.ucihar_model import AccelEncoder, GyroEncoder, HeadEncoder, MMOFLModel
import numpy as np


seeds = [0, 1, 2, 3]
methods = ["fm", "pm", "zf", "pmm"]


def fresh_model():
    return MMOFLModel([AccelEncoder(), GyroEncoder()], HeadEncoder(feat_dim=256, num_classes=6))


# Load test set once (any seed, test set is reproducible)
_, test_batch, _ = make_ucihar_clients("./UCI HAR Dataset", K=5, alpha=10, seed=0)


results = {m: [] for m in methods}
for s in seeds:
    for m in methods:
        path = f"results/models/{m}_seed{s}.pt"
        if not os.path.exists(path):
            continue
        model = fresh_model()
        model.load_state_dict(torch.load(path, map_location="cpu"))
        acc, _ = evaluate(model, test_batch)
        results[m].append(acc)
        print(f"{m} seed{s}: {acc:.4f}")


print("\nSummary:")
for m in methods:
    accs = results[m]
    if accs:
        import statistics
        print(f"{m.upper():<6} mean={statistics.mean(accs):.4f}  std={statistics.pstdev(accs):.4f}  n={len(accs)}")
