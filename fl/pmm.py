import torch


def compute_local_prototypes(model, batch):
    """OPC: Online Prototypes Construction, local step (eq. 15).

    Returns {(m, c): tensor(128,)} — encoder output features, not raw inputs.
    """
    was_training = model.training
    model.eval()
    protos = {}
    modality_data, labels = batch

    with torch.no_grad():
        for m, encoder in enumerate(model.encoders):
            features = encoder(modality_data[m])  # (N, 128)
            for c in labels.unique():
                mask = labels == c
                protos[(m, c.item())] = features[mask].mean(0)

    model.train(was_training)
    return protos


def get_proto_features(labels, missing_m, global_prototypes, feat_dim=128):
    """OPS: build the substitution feature matrix for one missing modality (eq. 19).

    Returns a (N, feat_dim) tensor where each row is the persistent global
    prototype for the sample's class, ready to be concatenated alongside the
    features from the available encoders.  This operates at the *feature*
    level — the missing encoder is never called.
    """
    device = labels.device
    zero = torch.zeros(feat_dim, device=device)
    rows = [
        global_prototypes.get((missing_m, y.item()), zero)
        for y in labels
    ]
    return torch.stack(rows)   # (N, feat_dim)
