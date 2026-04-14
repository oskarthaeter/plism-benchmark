"""Module for cosine similarity metric."""

import numpy as np

from plismbench.metrics.base import BasePlismMetric


class CosineSimilarity(BasePlismMetric):
    """Cosine similarity metric."""

    def __init__(self, device: str, use_mixed_precision: bool = True):
        super().__init__(device, use_mixed_precision)

    def compute_metric(self, matrix_a, matrix_b):
        """Compute cosine similarity metric."""
        if matrix_a.shape[0] != matrix_b.shape[0]:
            raise ValueError(
                f"Number of tiles must match. Got {matrix_a.shape[0]} and {matrix_b.shape[0]}."
            )

        if self.device == "gpu":
            import torch

            ta = torch.from_numpy(matrix_a).cuda()
            tb = torch.from_numpy(matrix_b).cuda()
            if self.use_mixed_precision:
                ta = ta.to(torch.float16)
                tb = tb.to(torch.float16)
            dot_product_ab = torch.matmul(ta, tb.T)
            norm_a = torch.linalg.norm(ta.float(), dim=1, keepdim=True)
            norm_b = torch.linalg.norm(tb.float(), dim=1, keepdim=True)
            cosine_ab = dot_product_ab.float() / (norm_a * norm_b.T)
            return float(torch.diag(cosine_ab).mean().item())
        else:
            matrix_a = matrix_a.astype(np.float16) if self.use_mixed_precision else matrix_a
            matrix_b = matrix_b.astype(np.float16) if self.use_mixed_precision else matrix_b
            dot_product_ab = np.matmul(matrix_a, matrix_b.T)
            norm_a = np.linalg.norm(matrix_a, axis=1, keepdims=True)
            norm_b = np.linalg.norm(matrix_b, axis=1, keepdims=True)
            cosine_ab = dot_product_ab / (norm_a * norm_b.T)
            return float(np.diag(cosine_ab).mean())
