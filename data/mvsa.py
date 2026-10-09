"""
MVSA-Single loader.

Source: http://mcrlab.net/research/mvsa-sentiment-analysis-on-multi-view-social-data/
  - 5,129 image-text pairs, 3 sentiment classes (0=negative, 1=neutral, 2=positive)
  - Download from the official site (requires registration) or use the
    Hugging Face mirror if available: https://huggingface.co/datasets/Nicolem/MVSA_Single
  - Modalities: raw image (JPEG) + raw text (tweet string)

Paper settings (MVSA-Single):
    K=5, initial 1500 samples/client, local buffer 800, 20 added/removed per round

CAVEAT: I do not have the actual dataset files, so this loader is written
against the commonly-documented MVSA-Single layout:

    MVSA_Single/
      data/
        1.jpg
        1.txt
        2.jpg
        2.txt
        ...
      labelResultAll.txt   # "ID text_label image_label", e.g. "1 positive negative"

MVSA-Single has ONE annotation per pair (unlike MVSA-Multiple's 3-annotator
majority vote), but the label file still lists a text label and an image
label separately, and standard practice keeps only pairs where they agree,
using that agreed label as the ground truth. VERIFY THIS MATCHES YOUR
DOWNLOADED FILES before trusting results — dataset mirrors sometimes
repackage this differently (e.g. a single combined label per row).
"""

import os
import numpy as np
from PIL import Image

from .ucihar import dirichlet_split  # reuse the same Dirichlet split logic

LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}


def load_mvsa(base_path, image_size=64, max_samples=None):
    """
    Load MVSA-Single image-text pairs and sentiment labels.

    Returns:
        images: (N, image_size, image_size, 3) uint8 array
        texts:  list[str], length N, raw tweet text
        labels: (N,) int array, 0/1/2
    """
    label_file = os.path.join(base_path, "labelResultAll.txt")
    if not os.path.exists(label_file):
        raise FileNotFoundError(
            f"Expected label file not found at {label_file}. "
            "Check your MVSA-Single download's directory layout and "
            "adjust load_mvsa() to match — see the module docstring."
        )

    images, texts, labels = [], [], []
    with open(label_file, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    header = lines[0]
    rows = lines[1:] if "text" in header.lower() or "ID" in header else lines

    for line in rows:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        sample_id, text_label, image_label = parts[0], parts[1], parts[2]

        if text_label != image_label:
            continue  # keep only agreed-label pairs (see caveat above)
        if text_label not in LABEL_MAP:
            continue

        img_path = os.path.join(base_path, "data", f"{sample_id}.jpg")
        txt_path = os.path.join(base_path, "data", f"{sample_id}.txt")
        if not (os.path.exists(img_path) and os.path.exists(txt_path)):
            continue

        img = Image.open(img_path).convert("RGB").resize((image_size, image_size))
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as tf:
            text = tf.read().strip()

        images.append(np.array(img, dtype=np.uint8))
        texts.append(text)
        labels.append(LABEL_MAP[text_label])

        if max_samples and len(labels) >= max_samples:
            break

    if not labels:
        raise ValueError(
            "Parsed 0 samples from labelResultAll.txt — the label file "
            "format likely differs from what this loader expects. Open "
            "the file and adjust the parsing above accordingly."
        )

    return np.stack(images), texts, np.array(labels)


def build_tokenizer(texts, vocab_size=10000):
    """
    Minimal whitespace tokenizer + fixed vocab, sufficient to feed
    models.mvsa_model.TextEncoder (nn.Embedding(vocab_size, ...)).
    Swap for a proper tokenizer (e.g. HF `transformers`) for real results.
    """
    from collections import Counter
    counter = Counter()
    for t in texts:
        counter.update(t.lower().split())

    vocab = {"<pad>": 0, "<unk>": 1}
    for word, _ in counter.most_common(vocab_size - len(vocab)):
        vocab[word] = len(vocab)

    def encode(text, max_len=32):
        ids = [vocab.get(w, 1) for w in text.lower().split()][:max_len]
        ids += [0] * (max_len - len(ids))
        return ids

    return vocab, encode


def build_client_splits(images, text_ids, labels, K=5, alpha=1.0,
                         initial_per_client=1500, seed=None):
    """
    Dirichlet-partition MVSA-Single across K clients.

    images:   tensor or array of shape (N, 3, H, W) float
    text_ids: tensor or array of shape (N, seq_len) int
    labels:   (N,) int array, 0/1/2

    Returns a list of K dicts:
        {
          'initial': list[(modality_list, label)],
          'long_term': list[(modality_list, label)],
        }
    where modality_list = [image_tensor (3, H, W), text_ids_tensor (seq_len,)]
    """
    if seed is not None:
        np.random.seed(seed)

    num_classes = int(np.max(labels)) + 1
    client_indices = dirichlet_split(labels, K, alpha, num_classes)

    import torch
    clients = []
    for idx_list in client_indices:
        idx = np.array(idx_list)
        np.random.shuffle(idx)

        init_idx = idx[:initial_per_client]
        rest_idx = idx[initial_per_client:]

        def to_samples(indices):
            out = []
            for i in indices:
                img = images[i]
                txt = text_ids[i]
                # Convert to plain tensors so StreamingBuffer can collate them
                if not isinstance(img, torch.Tensor):
                    img = torch.tensor(np.asarray(img), dtype=torch.float32)
                if not isinstance(txt, torch.Tensor):
                    txt = torch.tensor(np.asarray(txt), dtype=torch.long)
                out.append(([img, txt], int(labels[i])))
            return out

        clients.append({
            'initial': to_samples(init_idx),
            'long_term': to_samples(rest_idx),
        })

    return clients
