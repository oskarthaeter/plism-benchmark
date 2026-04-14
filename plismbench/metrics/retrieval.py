"""Module for retrieval metrics."""

import numpy as np

from plismbench.metrics.base import BasePlismMetric


class TopkAccuracy(BasePlismMetric):
    """Top-k accuracy."""

    def __init__(
        self,
        device: str,
        use_mixed_precision: bool = True,
        k: list[int] | None = None,
    ):
        super().__init__(device, use_mixed_precision)
        self.k = [1, 3, 5, 10] if k is None else k

    def compute_metric(self, matrix_a, matrix_b):
        """Compute top-k accuracy metric."""
        if matrix_a.shape[0] != matrix_b.shape[0]:
            raise ValueError(
                f"Number of tiles must match. Got {matrix_a.shape[0]} and {matrix_b.shape[0]}."
            )

        matrix_ab = np.concatenate([matrix_a, matrix_b], axis=0)
        n_tiles = matrix_ab.shape[0] // 2

        if self.use_mixed_precision:
            matrix_ab = matrix_ab.astype(np.float16)

        kmax = max(self.k)

        if self.device == "gpu":
            import torch

            tab = torch.from_numpy(matrix_ab).cuda()
            dot_product_ab = torch.matmul(tab, tab.T)
            norm_ab = torch.linalg.norm(tab.float(), dim=1, keepdim=True)
            cosine_ab = dot_product_ab.float() / (norm_ab * norm_ab.T)

            # topk returns sorted descending; index 0 is self-match (cosine=1), skip it
            top_kmax_indices_ab = torch.topk(cosine_ab, kmax + 1, dim=1).indices[:, 1:]

            top_k_accuracies = []
            for k in self.k:
                top_k_indices_ab = top_kmax_indices_ab[:, :k]
                top_k_indices_a = top_k_indices_ab[:n_tiles]
                top_k_indices_b = top_k_indices_ab[n_tiles:]

                top_k_accs = []
                for i, top_k_indices in enumerate([top_k_indices_a, top_k_indices_b]):
                    other_slide_indices = (
                        torch.arange(n_tiles, 2 * n_tiles, device="cuda")
                        if i == 0
                        else torch.arange(0, n_tiles, device="cuda")
                    )
                    correct_matches = (
                        top_k_indices == other_slide_indices.unsqueeze(1)
                    ).any(dim=1).sum()
                    top_k_accs.append(float(correct_matches.item()) / n_tiles)

                top_k_accuracies.append(sum(top_k_accs) / 2)

            return np.array(top_k_accuracies)
        else:
            tab = self.ncp.asarray(matrix_ab)
            dot_product_ab = self.ncp.matmul(tab, tab.T)
            norm_ab = self.ncp.linalg.norm(tab, axis=1, keepdims=True)
            cosine_ab = dot_product_ab / (norm_ab * norm_ab.T)

            top_kmax_indices_ab = np.argpartition(
                -cosine_ab, range(1, kmax + 1), axis=1
            )[:, 1 : kmax + 1]

            top_k_accuracies = []
            for k in self.k:
                top_k_indices_ab = top_kmax_indices_ab[:, :k]
                top_k_indices_a = top_k_indices_ab[:n_tiles]
                top_k_indices_b = top_k_indices_ab[n_tiles:]

                top_k_accs = []
                for i, top_k_indices in enumerate([top_k_indices_a, top_k_indices_b]):
                    other_slide_indices = (
                        self.ncp.arange(n_tiles, 2 * n_tiles)
                        if i == 0
                        else self.ncp.arange(0, n_tiles)
                    )
                    correct_matches = self.ncp.sum(
                        self.ncp.any(top_k_indices == other_slide_indices[:, None], axis=1)
                    )
                    top_k_accs.append(float(correct_matches) / n_tiles)

                top_k_accuracies.append(sum(top_k_accs) / 2)

            return np.array(top_k_accuracies)
