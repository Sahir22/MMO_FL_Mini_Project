# MMO-FL + PMM Replication

Implementation for replicating:

**"Multimodal Online Federated Learning With Modality Missing in IoT"**
— Wang et al., IEEE TMC 2026

## Core contributions replicated
1. **MMO-FL training loop** — online federated learning with streaming data
2. **PMM algorithm** — Online Prototypes Construction (OPC) + Online Prototypes Substitution (OPS)
3. **Experiments** on UCI-HAR and MVSA-Single with FM / PM / ZF / PMM benchmarks

## Project structure
```
mmo_fl/
├── data/
│   ├── ucihar.py        # loader + Dirichlet split
│   └── mvsa.py          # image/text loader + split
├── models/
│   ├── ucihar_model.py  # AccelEncoder, GyroEncoder, Head, MMOFLModel
│   └── mvsa_model.py    # ImageEncoder, TextEncoder
├── fl/
│   ├── client.py        # StreamingBuffer + local OGD update
│   ├── server.py        # FedAvg + prototype aggregation
│   └── pmm.py           # OPC + OPS logic
├── baselines/
│   └── benchmarks.py    # FM, PM, ZF, PMM wrappers
├── train.py             # single/multi-seed training runner
├── compare.py           # head-to-head FM/PM/ZF/PMM comparison (Tables II/III)
├── plot.py              # accuracy-vs-rounds figure generator (Figs 3-8)
├── ablation/
│   └── run_ablations.py # four ablation studies
├── setup_env.sh
└── requirements.txt
```

## Setup
```bash
python3 -m venv mmo-fl
source mmo-fl/bin/activate
pip install -r requirements.txt
```

## Datasets

**UCI-HAR**
- [UCI ML Repository](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
- Download and extract `UCI HAR Dataset.zip` to `./UCI HAR Dataset/`

**MVSA-Single**
- [Hugging Face mirror](https://huggingface.co/datasets/Nicolem/MVSA_Single)
- Extract so the layout is `MVSA_Single/data/*.jpg`, `MVSA_Single/data/*.txt`, `MVSA_Single/labelResultAll.txt`
- MVSA logic is integrated, but results are not available for this yet

---

## How to run

### 1. Smoke test (no dataset needed)
```bash
python3 train.py --dataset synthetic --rounds 5 --clients 3
```

### 2. Single training run — UCI-HAR (replicates main result)
```bash
python3 train.py --dataset ucihar \
    --data-path "./UCI HAR Dataset" \
    --rounds 100 --clients 5 --alpha 1 --lambda-missing 0.5 --E 1
```

### 3. Head-to-head comparison (FM / PM / ZF / PMM) — Table II/III
```bash
# UCI-HAR
python3 compare.py --dataset ucihar \
    --data-path "./UCI HAR Dataset" \
    --rounds 100 --clients 5 --alpha 1 --lambda-missing 0.5 \
    --seeds 10

# MVSA-Single
python3 compare.py --dataset mvsa \
    --data-path ./MVSA_Single \
    --rounds 120 --clients 5 --alpha 1 --lambda-missing 0.5 \
    --seeds 10
```
Output: `results/compare_*.csv` (per-round) + `results/compare_*_summary.txt` (table)

### 4. Plot accuracy-vs-rounds curves (Figs 3–4)
```bash
python3 plot.py \
    --csv results/compare_ucihar_T100_lam0.5_alpha1.0.csv \
    --out results/fig_compare.png
```

### 5. Ablation studies (Figs 5–8)
```bash
# Missing rate (Fig. 5)
python3 ablation/run_ablations.py --study missing_rate \
    --dataset ucihar --data-path "./UCI HAR Dataset" \
    --rounds 100 --seeds 10

# Non-IID level (Fig. 6)
python3 ablation/run_ablations.py --study noniid \
    --dataset ucihar --data-path "./UCI HAR Dataset" \
    --rounds 100 --seeds 10

# Quantized upload (Fig. 7)
python3 ablation/run_ablations.py --study quant \
    --dataset ucihar --data-path "./UCI HAR Dataset" \
    --rounds 100 --seeds 10

# Delayed OPC update (Fig. 8)
python3 ablation/run_ablations.py --study delay \
    --dataset ucihar --data-path "./UCI HAR Dataset" \
    --rounds 100 --seeds 10

# Plot any ablation CSV
python3 plot.py \
    --csv ablation/results/ablation_missing_rate.csv \
    --hue lambda \
    --out results/fig5_missing_rate.png
```

### 6. Multi-seed (paper uses 10 seeds)
Add `--seeds 10` to any `train.py`, `compare.py`, or ablation command.

---

## Hyperparameters

**UCI-HAR**

| Parameter | Value |
|---|---|
| K (clients) | 5 |
| T (global rounds) | 100 |
| α (Non-IID) | 1, 5, 10 |
| λ (missing rate) | 0.3, 0.5, 0.7 |
| LR η | 0.1, decay 0.95/round → min 0.001 |
| Initial samples/client | 2000 |
| Buffer size | 500 |
| New samples/round | 20 |

**MVSA-Single**

| Parameter | Value |
|---|---|
| K (clients) | 5 |
| T (global rounds) | 120 |
| α (Non-IID) | 1 |
| λ (missing rate) | 0.5 |
| LR η | 0.01, decay 0.99 → min 0.001 |
| Initial samples/client | 1500 |
| Buffer size | 800 |
| New samples/round | 20 |
