from collections import deque


import numpy as np
import torch
import torch.nn as nn




class DataPool:
   """
   Backs StreamingBuffer.long_term_source. Wraps a client's remaining
   (post-initial) samples and supports repeated `.sample(n)` calls,
   cycling with a reshuffle once exhausted so a client never runs out of
   "new" data over T rounds even if long_term is smaller than T * n_new.
   """


   def __init__(self, samples, seed=None):
       # samples: list of (modality_list, label)
       self.samples = list(samples)
       self.rng = np.random.RandomState(seed)
       self._order = self.rng.permutation(len(self.samples))
       self._cursor = 0


   def sample(self, n):
       if len(self.samples) == 0:
           return []
       out = []
       for _ in range(n):
           if self._cursor >= len(self._order):
               self._order = self.rng.permutation(len(self.samples))
               self._cursor = 0
           out.append(self.samples[self._order[self._cursor]])
           self._cursor += 1
       return out




class StreamingBuffer:
   """Sliding-window dataset that evolves each round for a single client."""


   def __init__(self, initial_data, max_size, long_term_source=None):
       # initial_data: list of (modality_list, label)
       self.buffer = deque(initial_data, maxlen=max_size)
       # long_term_source: DataPool (or anything with .sample(n))
       self.long_term_source = long_term_source


   def update(self, n_new=20):
       """Add n_new new samples, drop oldest (FIFO via deque maxlen)."""
       if self.long_term_source is None:
           return
       new_samples = self.long_term_source.sample(n_new)
       for s in new_samples:
           self.buffer.append(s)  # maxlen handles eviction


   def get_all(self):
       """
       Collate the current buffer into batched (modality_data, labels).


       Each sample is (modality_list, label).  Supports two cases:
         - modality_list contains numpy arrays (UCI-HAR): stacked with np.stack
         - modality_list contains torch tensors (MVSA): stacked with torch.stack
       """
       if len(self.buffer) == 0:
           raise ValueError("StreamingBuffer is empty — call update() first.")


       num_modalities = len(self.buffer[0][0])
       modality_data = []
       for m in range(num_modalities):
           first = self.buffer[0][0][m]
           if isinstance(first, torch.Tensor):
               stacked = torch.stack([sample[0][m] for sample in self.buffer])
           else:
               stacked = torch.tensor(
                   np.stack([np.asarray(sample[0][m]) for sample in self.buffer]),
                   dtype=torch.float32,
               )
           modality_data.append(stacked)


       labels = torch.tensor(
           [sample[1] for sample in self.buffer], dtype=torch.long
       )
       return modality_data, labels




class Client:
   """Bundles a client's local model with its StreamingBuffer."""


   def __init__(self, client_id, model, buffer: StreamingBuffer):
       self.id = client_id
       self.model = model
       self.buffer = buffer




def do_local_update(model, batch, E, lr, missing=None, proto_features=None,
                   batch_size=64):
   """E epochs of mini-batch SGD on a client's local batch.


   When `missing` is set and `proto_features` is provided, OPS is applied:
   model.forward_ops() is called, which injects the precomputed prototype
   feature for the missing modality at the *feature* level (encoder not
   called).  The missing encoder's parameters receive no gradient.


   When `missing` is set but `proto_features` is None (cold start or PM
   baseline), the missing encoder's input is zeroed and its gradients are
   suppressed (eq. 6 / PM behaviour).
   """
   model.train()
   optimizer = torch.optim.SGD(model.parameters(), lr=lr)
   criterion = nn.CrossEntropyLoss()


   modality_data, labels = batch
   N = labels.shape[0]
   total_loss = 0.0
   n_steps = 0


   for _ in range(E):
       perm = torch.randperm(N)
       for start in range(0, N, batch_size):
           idx = perm[start:start + batch_size]
           mb_data = [x[idx] for x in modality_data]
           mb_labels = labels[idx]


           optimizer.zero_grad()


           if missing is not None and proto_features is not None:
               # OPS path: prototype injected at feature level
               logits = model.forward_ops(mb_data, missing, proto_features[idx])
           else:
               # PM / cold-start path: zero out raw input, suppress encoder grad
               if missing is not None:
                   mb_data = [x if i != missing else torch.zeros_like(x)
                              for i, x in enumerate(mb_data)]
               logits = model(mb_data)


           loss = criterion(logits, mb_labels)
           total_loss += loss.item()
           n_steps += 1
           loss.backward()


           # Suppress gradients for missing encoder in both missing paths
           if missing is not None:
               for p in model.encoders[missing].parameters():
                   if p.grad is not None:
                       p.grad.zero_()

           optimizer.step()


   return total_loss / max(n_steps, 1)