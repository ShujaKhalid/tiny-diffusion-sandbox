import torch
from torch.utils.data import DataLoader
from model import MLP
from noise_scheduler import NoiseScheduler

import numpy as np
from config import Config

from utils import get_dataset


def train(cfg, ds, dl, model, opt, criterion, ns):
    for epoch in range(cfg.epochs):
        model.train()
        train_loss = 0.0
        for batch in dl:
            batch = batch[0]
            t = np.random.randint(0, cfg.timesteps)
            eps_target = torch.randn(batch.shape).cuda()
            x_new = ns.add_noise(batch, t, eps_target)
            eps_pred = model(x_new, t)
            loss = criterion(eps_pred, eps_target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()
        train_loss /= (len(ds))
        print("epoch: {} | loss: {}".format(epoch+1, train_loss))

    torch.save(model.state_dict(), cfg.log_dir + "params.pt")
    print(f"saving model to {cfg.log_dir}")


def main(cfg):
    # Preliminaries
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # Initialize the dataloader
    ds = get_dataset(cfg)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    # Initialize the model
    model = MLP(cfg).cuda()

    # Initialize the optimizer
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = torch.nn.MSELoss(reduction="sum")

    # Initialize the noise_scheduler
    ns = NoiseScheduler(cfg)

    # Train the model
    train(cfg, ds, dl, model, opt, criterion, ns)

    print("Training completed!")


if __name__ == "__main__":
    cfg = Config()
    main(cfg)