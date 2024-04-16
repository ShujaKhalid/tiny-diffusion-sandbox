import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from model import Model
from noise_scheduler import NoiseScheduler

import pandas as pd
import numpy as np
from utils import Config

def get_dataset(cfg):
    fn = cfg.csv_file
    df = pd.read_csv(fn)
    x = df["x"]
    y = df["y"]

    # TODO: process the data as needed

    x = torch.from_numpy(np.stack((x, y), axis=1)).float().cuda()

    return TensorDataset(x)


def train(cfg, dl, model, opt, criterion, ns):
    for epoch in range(cfg.epochs):
        train_loss = 0.0
        for batch in dl:
            opt.zero_grad()
            batch = batch[0]
            t = np.random.randint(0, cfg.timesteps)
            eps_target = torch.randn_like(batch)
            x_new = ns.add_noise(batch, t, eps_target)
            eps_pred = model(x_new, t)
            loss = criterion(eps_pred, eps_target)
            loss.backward()
            opt.step()
            train_loss += loss.item()
        train_loss /= len(dl)
        print("epoch: {} | loss: {}".format(epoch, train_loss))

    torch.save(model.state_dict(), cfg.log_dir + "params.pt")
    print(f"saving model to {cfg.log_dir}")

def main(cfg):
    # Preliminaries
    torch.manual_seed(cfg.seed)

    # Initialize the dataloader
    ds = get_dataset(cfg)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    # Initialize the model
    model = Model(cfg).cuda()
    model.train()

    # Initialize the optimizer
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = torch.nn.MSELoss()

    # Initialize the noise_scheduler
    ns = NoiseScheduler(cfg)

    # Train the model
    train(cfg, dl, model, opt, criterion, ns)

    print("Training completed!")


if __name__ == "__main__":

    # Extract configuration parameters here
    cfg = {
        "seed": 0,

        # data
        "csv_file": "assets/simple/cat.csv",
    
        # opt_params
        "epochs": 5,
        "batch_size": 32,
        "lr": 1e-3,

        # model_params
        "model": "MLP", # denoiser
        "input_dim": 3,
        "hidden_dim": 64,
        "output_dim": 2,
        "hidden_layers": 3,

        # output_params
        "log_dir": "./logs/",

        # noise_scheduler
        "beta_start": 1e-5,
        "beta_end": 1e-2,
        "timesteps": 50,
    }

    cfg = Config(cfg)

    main(cfg)