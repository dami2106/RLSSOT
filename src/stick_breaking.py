import torch
import torch.nn as nn


class StickBreakingProcess(nn.Module):
    """Variational stick-breaking process for inferring the number of skills.

    This module maintains Beta parameters for a truncated stick-breaking
    representation of a Dirichlet Process prior. The sufficient statistics are
    updated from soft cluster assignments produced by OT, allowing the model to
    activate or deactivate clusters based on the data instead of a fixed K.
    """

    def __init__(self, max_clusters, concentration=1.0, threshold=1e-3, momentum=0.5):
        super().__init__()
        self.max_clusters = max_clusters
        self.concentration = concentration
        self.threshold = threshold
        self.momentum = momentum

        self.register_buffer('alpha', torch.ones(max_clusters))
        self.register_buffer('beta', torch.ones(max_clusters) * concentration)

    def expected_v(self):
        return self.alpha / (self.alpha + self.beta + 1e-12)

    def expected_weights(self):
        v = self.expected_v()
        stick_remaining = torch.cumprod(
            torch.cat([
                torch.ones(1, device=v.device),
                1.0 - v + 1e-12
            ]),
            dim=0
        )
        stick_remaining = stick_remaining[:-1]
        weights = v * stick_remaining
        if weights.numel() > 0:
            tail_mass = torch.clamp(1.0 - weights.sum(), min=0.0)
            weights = weights.clone()
            weights[-1] = weights[-1] + tail_mass
        weights = torch.clamp(weights, min=0.0)
        return weights

    def active_count(self):
        weights = self.expected_weights()
        total = weights.sum()
        if total <= 0:
            return 1
        normalized = weights / total
        mask = normalized > self.threshold
        if mask.any():
            active = int(mask.nonzero(as_tuple=False).max().item()) + 1
        else:
            active = 1
        cumulative = torch.cumsum(normalized, dim=0)
        if active < self.max_clusters and cumulative[active - 1] < 1 - self.threshold:
            active += 1
        return min(self.max_clusters, max(1, active))

    def update(self, responsibilities, mask=None):
        if responsibilities.numel() == 0:
            return
        if mask is not None:
            mask = mask.to(responsibilities.dtype)
            responsibilities = responsibilities * mask.unsqueeze(-1)

        dims = list(range(responsibilities.dim() - 1))
        counts = responsibilities.sum(dim=dims)

        if counts.sum() <= 0:
            return

        K = counts.shape[0]
        tail_counts = torch.cumsum(torch.flip(counts, dims=[0]), dim=0)
        tail_counts = torch.flip(tail_counts, dims=[0]) - counts

        new_alpha = 1.0 + counts
        new_beta = self.concentration + tail_counts

        with torch.no_grad():
            if self.momentum is not None:
                m = self.momentum
                self.alpha[:K] = m * self.alpha[:K] + (1 - m) * new_alpha
                self.beta[:K] = m * self.beta[:K] + (1 - m) * new_beta
            else:
                self.alpha[:K] = new_alpha
                self.beta[:K] = new_beta
            # Reset unused sticks to the prior to avoid stale values
            if K < self.max_clusters:
                self.alpha[K:] = 1.0
                self.beta[K:] = self.concentration

