import torch


def fedavg(global_model, local_states):
    """Federated averaging of client model state dicts (eq. 7)."""
    avg_state = {}
    for key in local_states[0]:
        avg_state[key] = torch.stack([s[key].float() for s in local_states]).mean(0)
    global_model.load_state_dict(avg_state)
    return global_model


def update_global_prototypes(prototypes, prototype_counts, new_local_prototypes):
    """Server-side persistent prototype update (OPC, eq. 16-17)."""
    for (m, c), proto_list in new_local_prototypes.items():
        temporal_proto = torch.stack(proto_list).mean(0)
        if (m, c) in prototypes:
            n = prototype_counts[(m, c)]
            prototypes[(m, c)] = (n * prototypes[(m, c)] + temporal_proto) / (n + 1)
            prototype_counts[(m, c)] = n + 1
        else:
            prototypes[(m, c)] = temporal_proto
            prototype_counts[(m, c)] = 1
    return prototypes, prototype_counts
