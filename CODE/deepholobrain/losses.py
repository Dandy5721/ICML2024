"""Auxiliary embedding losses used alongside the classification/regression criterion."""

import torch


def cosine_similarity(input):
    return input @ input.transpose(-2, -1) * 0.5 + 0.5


def similarity_loss(input, targets, alpha=0):
    """Pull same-class samples together / push different-class samples apart by cosine similarity."""
    similarity = cosine_similarity(input)
    same_class = targets.unsqueeze(-2) == targets.unsqueeze(-2).transpose(-2, -1)
    loss = (1 - similarity) * same_class + torch.clamp(similarity - alpha, min=0) * (~same_class)
    return torch.mean(loss)


def distance_loss(inputs, targets):
    """Pull same-class SPD matrices together / push different-class ones apart in tangent space."""
    same_class = targets.unsqueeze(0) == targets.unsqueeze(0).T
    L, U = torch.linalg.eigh(inputs, UPLO='U')
    log_X = U @ torch.diag_embed(L.log()) @ U.transpose(-2, -1)

    pair_dist = torch.norm(
        log_X.unsqueeze(-4) - log_X.unsqueeze(-3) + 1e-7, p='fro', dim=(-2, -1)
    )
    loss = pair_dist * same_class - pair_dist * (~same_class)
    return loss.mean()
