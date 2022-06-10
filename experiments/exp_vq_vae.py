import numpy as np
from einops import rearrange
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from experiments.exp_base import *
from encoder_decoders.vq_vae_encdec import VQVAEEncoder, VQVAEDecoder
from vector_quantize_pytorch import VectorQuantize


class ExpVQVAE(ExpBase):
    def __init__(self,
                 config: dict = None,
                 n_train_samples: int = None):
        super().__init__()
        dim = config['VQ-VAE']['dim']
        codebook_size = config['VQ-VAE']['codebook_size']
        decay = config['VQ-VAE']['decay']
        commitment_weight = config['VQ-VAE']['commitment_weight']

        self.encoder = VQVAEEncoder(dim)
        self.decoder = VQVAEDecoder(dim)
        self.vq_model = VectorQuantize(dim=dim,
                                       codebook_size=codebook_size,          # codebook size
                                       decay=decay,                          # the exponential moving average decay, lower means the dictionary will change faster
                                       commitment_weight=commitment_weight   # the weight on the commitment loss
                                       )
        self.config = config
        self.T_max = config['trainer_params']['max_epochs'] * (np.ceil(n_train_samples / config['dataset']['batch_size']) )

    def forward(self, ):
        pass

    def training_step(self, batch, batch_idx):
        x, y = batch  # x: (B, C, H, W)

        # forward
        z = self.encoder(x)
        h, w = z.shape[2], z.shape[3]
        z = rearrange(z, 'b c h w -> b (h w) c')
        z_q, indices, commit_loss, perplexity = self.vq_model(z)
        z_q = rearrange(z, 'b (h w) c -> b c h w', h=h, w=w)
        xhat = self.decoder(z_q)
        recons_loss = F.mse_loss(xhat, x)

        # loss
        loss = recons_loss + commit_loss

        # lr scheduler
        sch = self.lr_schedulers()
        sch.step()

        # log
        loss_hist = {'loss': loss,
                     'recons_loss': recons_loss,
                     'commit_loss': commit_loss,
                     'perplexity': perplexity}

        detach_the_unnecessary(loss_hist)
        return loss_hist

    def validation_step(self, batch, batch_idx):
        x, y = batch  # x: (B, C, H, W)

        # forward
        z = self.encoder(x)
        h, w = z.shape[2], z.shape[3]
        z = rearrange(z, 'b c h w -> b (h w) c')
        z_q, indices, commit_loss, perplexity = self.vq_model(z)
        z_q = rearrange(z, 'b (h w) c -> b c h w', h=h, w=w)
        xhat = self.decoder(z_q)
        recons_loss = F.mse_loss(xhat, x)

        # loss
        loss = recons_loss + commit_loss

        # log
        loss_hist = {'loss': loss,
                     'recons_loss': recons_loss,
                     'commit_loss': commit_loss,
                     'perplexity': perplexity}

        detach_the_unnecessary(loss_hist)
        return loss_hist

    def configure_optimizers(self):
        opt = torch.optim.AdamW([{'params': self.encoder.parameters(), 'lr': self.config['exp_params']['LR']},
                                 {'params': self.decoder.parameters(), 'lr': self.config['exp_params']['LR']},
                                 {'params': self.vq_model.parameters(), 'lr': self.config['exp_params']['LR']},
                                 ],
                                weight_decay=self.config['exp_params']['weight_decay'])
        return {'optimizer': opt, 'lr_scheduler': CosineAnnealingLR(opt, self.T_max)}
