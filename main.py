from argparse import ArgumentParser

import wandb
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
    wandb.finish()
