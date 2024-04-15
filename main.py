import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from model import Model
from noise_scheduler import NoiseScheduler

import pandas as pd
import numpy as np


def get_dataset(cfg):
    fn = cfg.csv_file
    df = pd.read_csv(fn)
    x = df["x"]
    y = df["y"]

    # TODO: process the data as needed

    x = torch.from_numpy(np.stack((x, y), axis=1))

    return TensorDataset(x)

def train(cfg):

    for epoch in range(cfg.epochs):



def main(cfg):
    # Preliminaries
    torch.set_random_seed(cfg.seed)

    # Initialize the dataloader
    ds = Dataset(cfg)
    dl = DataLoader(ds)

    # Initialize the model
    model = Model(cfg)
    model.train()

    # Initialize the optimizer
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = torch.nn.MSELoss()

    # Initialize the noise_scheduler
    ns = NoiseScheduler(cfg)

    # Train the model
    train(cfg, dl, model)

    print("Training completed!")


if __name__ == "__main__":

    # Extract configuration parameters here
    cfg = {
        "seed": 0,

        # data
        "csv_file": "assets/simple/cat.csv",
    
        # opt_params
        "epochs": 10,
        "batch_size": 32,
        "lr": 1e-3,

        # model_params
        "model": "MLP", # denoiser
        "input_dims": 3,
        "hidden_dim": 8,
        "output_dim": 2,
        "hidden_layers": 3,

        # output_params
        "log_dir": "./logs/",

        # noise_scheduler
        "beta_start": 1e-5,
        "beta_end": 1e-2,
        "timesteps": 50,
    }

    main(cfg)