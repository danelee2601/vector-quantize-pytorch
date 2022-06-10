import numpy as np
from einops import rearrange
import torch.nn.functional as F

from experiments.exp_vq_vae import ExpVQVAE
from encoder_decoders.vq_vae_encdec import VQVAEEncoder, VQVAEDecoder
from vector_quantize_pytorch import ResidualVQ
from experiments.exp_base import *


class ExpRQVAE(ExpVQVAE):
    def __init__(self,
                 config: dict = None,
                 n_train_samples: int = None):
        super().__init__(config, n_train_samples)
        dim = config['RQ-VAE']['dim']
        codebook_size = config['RQ-VAE']['codebook_size']
        decay = config['RQ-VAE']['decay']
        num_quantizers = config['RQ-VAE']['num_quantizers']

        self.encoder = VQVAEEncoder(dim)
        self.decoder = VQVAEDecoder(dim)
        self.vq_model = ResidualVQ(
            dim=dim,
            codebook_size=codebook_size,    # codebook size
            decay=decay,                    # the exponential moving average decay, lower means the dictionary will change faster
            num_quantizers=num_quantizers,  # specify number of quantizers
        )
        self.config = config
        self.T_max = config['trainer_params']['max_epochs'] * (
            np.ceil(n_train_samples / config['dataset']['batch_size']))

    def training_step(self, batch, batch_idx):
        x, y = batch  # x: (B, C, H, W)

        # forward
        z = self.encoder(x)
        h, w = z.shape[2], z.shape[3]
        z = rearrange(z, 'b c h w -> b (h w) c')
        z_q, indices, commit_loss, perplexity = self.vq_model(z)
        commit_loss = commit_loss.sum()
        z_q = rearrange(z, 'b (h w) c -> b c h w', h=h, w=w)
        xhat = self.decoder(z_q)
        recons_loss = F.mse_loss(xhat, x)

        # loss
        loss = recons_loss + commit_loss

        # lr scheduler
        sch = self.lr_schedulers()
        sch.step()

        # log
        loss_hist = {'loss': loss, 'recons_loss': recons_loss, 'commit_loss': commit_loss, 'perplexity': perplexity}

        detach_the_unnecessary(loss_hist)
        return loss_hist

    def validation_step(self, batch, batch_idx):
        x, y = batch  # x: (B, C, H, W)

        # forward
        z = self.encoder(x)
        h, w = z.shape[2], z.shape[3]
        z = rearrange(z, 'b c h w -> b (h w) c')
        z_q, indices, commit_loss, perplexity = self.vq_model(z)
        commit_loss = commit_loss.sum()
        z_q = rearrange(z, 'b (h w) c -> b c h w', h=h, w=w)
        xhat = self.decoder(z_q)
        recons_loss = F.mse_loss(xhat, x)

        # loss
        loss = recons_loss + commit_loss

        # log
        loss_hist = {'loss': loss, 'recons_loss': recons_loss, 'commit_loss': commit_loss, 'perplexity': perplexity}

        detach_the_unnecessary(loss_hist)
        return loss_hist