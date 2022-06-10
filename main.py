from argparse import ArgumentParser

import einops
import wandb
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor

from utils import root_dir, load_yaml_param_settings
from preprocessing.build_data_pipeline import build_data_pipeline
from experiments import experiments


def load_args():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, help="Path to the config data  file.",
                        default=root_dir.joinpath('configs', 'config.yaml'))
    return parser.parse_args()


if __name__ == '__main__':
    # load config
    args = load_args()
    config = load_yaml_param_settings(args.config)

    # data pipeline
    train_data_loader, val_data_loader = build_data_pipeline(config)

    # fit
    train_exp = experiments[config['model_name']](config, len(train_data_loader.dataset))
    wandb_logger = WandbLogger(project='VQ', name=config['model_name'], config=config)
    trainer = pl.Trainer(logger=wandb_logger,
                         enable_checkpointing=False,
                         callbacks=[LearningRateMonitor(logging_interval='epoch')],
                         **config['trainer_params'])
    trainer.fit(train_exp,
                train_dataloaders=train_data_loader,
                val_dataloaders=val_data_loader)

    # test
    print('testing...')
    for batch in val_data_loader:
        x, y = batch  # x: (B, C, H, W)

        z = train_exp.encoder(x)
        h, w = z.shape[2], z.shape[3]
        z = einops.rearrange(z, 'b c h w -> b (h w) c')
        z_q, indices, commit_loss, perplexity = train_exp.vq_model(z)
        z_q = einops.rearrange(z, 'b (h w) c -> b c h w', h=h, w=w)
        xhat = train_exp.decoder(z_q)

        wandb.log({"x": wandb.Image(x), "xhat": wandb.Image(xhat)})
        break

    wandb.finish()
