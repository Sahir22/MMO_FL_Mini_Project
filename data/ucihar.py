"""
UCI-HAR loader + Dirichlet non-IID split.

Source: UCI ML Repository - Human Activity Recognition Using Smartphones
https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

- 10,299 samples, 30 participants, 6 activity classes
- Two modalities: body_acc_xyz (accelerometer) and body_gyro_xyz (gyroscope),
  each of shape (128, 3)
- Download and extract "UCI HAR Dataset.zip" before use.
"""

import numpy as np
import os
from scipy.stats import dirichlet


def load_ucihar(base_path):
    def read_signals(path, signal_names):
        return np.stack([
            np.loadtxt(os.path.join(path, f'{s}.txt'))
            for s in signal_names
        ], axis=-1)  # (N, 128, 3)

    data = {}
    for split in ['train', 'test']:
        accel = read_signals(
            f'{base_path}/{split}/Inertial Signals',
            [f'body_acc_x_{split}', f'body_acc_y_{split}', f'body_acc_z_{split}']
        )  # (N, 128, 3)
        gyro = read_signals(
            f'{base_path}/{split}/Inertial Signals',
            [f'body_gyro_x_{split}', f'body_gyro_y_{split}', f'body_gyro_z_{split}']
        )
        labels = np.loadtxt(f'{base_path}/{split}/y_{split}.txt').astype(int) - 1
        data[split] = (accel, gyro, labels)

    return data


def dirichlet_split(labels, K, alpha, num_classes):
    """Returns list of K index arrays."""
    class_indices = [np.where(labels == c)[0] for c in range(num_classes)]
    client_indices = [[] for _ in range(K)]

    for c_idx in class_indices:
        proportions = np.random.dirichlet([alpha] * K)
        proportions = (proportions * len(c_idx)).astype(int)
        # fix rounding
        proportions[-1] = len(c_idx) - proportions[:-1].sum()
        splits = np.split(np.random.permutation(c_idx), proportions.cumsum()[:-1])
        for k, s in enumerate(splits):
            client_indices[k].extend(s.tolist())

    return client_indices


# Paper settings (UCI-HAR):
#   K=5, initial 2000 samples/client, local buffer 500, 20 added/removed per round


def build_client_splits(accel, gyro, labels, K=5, alpha=1.0,
                         initial_per_client=2000, seed=None):
    """
    Full pipeline: Dirichlet-partition the training set across K clients,
    then split each client's indices into an `initial` pool (used to seed
    the StreamingBuffer) and a `long_term` pool (the reservoir that
    StreamingBuffer.update() samples new points from each round).

    Returns a list of dicts, one per client:
        {
          'initial': list[(modality_list, label)],
          'long_term': list[(modality_list, label)],
        }
    where modality_list = [accel_sample (128,3), gyro_sample (128,3)]
    """
    if seed is not None:
        np.random.seed(seed)

    num_classes = int(labels.max()) + 1
    client_indices = dirichlet_split(labels, K, alpha, num_classes)

    clients = []
    for idx_list in client_indices:
        idx = np.array(idx_list)
        np.random.shuffle(idx)

        init_idx = idx[:initial_per_client]
        rest_idx = idx[initial_per_client:]

        def to_samples(indices):
            out = []
            for i in indices:
                modality_list = [accel[i], gyro[i]]
                out.append((modality_list, int(labels[i])))
            return out

        clients.append({
            'initial': to_samples(init_idx),
            'long_term': to_samples(rest_idx),
        })

    return clients
