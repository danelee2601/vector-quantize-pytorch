from functools import partial
import torch
from torch import nn
from vector_quantize_pytorch.vector_quantize_pytorch import VectorQuantize


class ResidualVQ(nn.Module):
    """ Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf """
    def __init__(
        self,
        *,
        num_quantizers,
        shared_codebook = False,
        **kwargs
    ):
        super().__init__()
        self.codebook_size = kwargs.get('codebook_size', None)
        self.layers = nn.ModuleList([VectorQuantize(**kwargs) for _ in range(num_quantizers)])

        if not shared_codebook:
            return

        first_vq, *rest_vq = self.layers
        codebook = first_vq._codebook

        for vq in rest_vq:
            vq._codebook = codebook

    def forward(self, x):
        quantized_out = 0.
        residual = x

        all_losses = []
        all_indices = []
        embed_onehots = torch.zeros((x.shape[0] * x.shape[1], self.codebook_size)).to(x.device)
        for layer in self.layers:
            quantized, indices, loss, _ = layer(residual)
            embed_onehots += layer._codebook.embed_onehot
            residual = residual - quantized
            quantized_out = quantized_out + quantized

            all_indices.append(indices)
            all_losses.append(loss)
        embed_onehots /= len(self.layers)

        all_losses, all_indices = map(partial(torch.stack, dim = -1), (all_losses, all_indices))

        avg_probs = torch.mean(embed_onehots, dim=0)  # (K,)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized_out, all_indices, all_losses, perplexity
