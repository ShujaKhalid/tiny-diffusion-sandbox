import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from torch.utils.data import DataLoader, Dataset, TensorDataset

import torch
import numpy as np
import pandas as pd


def get_dataset(cfg, e=0.01):
    fn = cfg.csv_file
    df = pd.read_csv(fn)
    x = df["x"]
    y = df["y"]

    # TODO: process the data as needed
    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()

    indices = np.random.randint(0, len(df), size=cfg.dataset_size)
    x = x.iloc[indices]
    y = y.iloc[indices]

    x += np.random.normal(size=len(x)) * e
    y += np.random.normal(size=len(y)) * e

    X = np.stack((x, y), axis=1)
    X = TensorDataset(torch.from_numpy(X.astype(np.float32)).cuda())

    return X


def viz_sample(data, alpha: float = 0.3, figsize: int = 4, l: float = 2.5) -> None:
    x = data[:, 0]
    y = data[:, 1]
    plt.figure(figsize=(figsize, figsize))
    plt.scatter(x, y, s=3, alpha=alpha)
    plt.xlim(-l, l)
    plt.ylim(-l, l)
    plt.xticks([-l, 0, l])
    plt.yticks([-l, 0, l])
    plt.show()


def viz_samples(samples, alpha: float = 0.3, figsize: int = 4, l: float = 2.5) -> None:
    fig, ax = plt.subplots(figsize=(figsize, figsize))
    ax.set_xlim(-l, l)
    ax.set_ylim(-l, l)
    ax.set_xticks([-l, 0, l])
    ax.set_yticks([-l, 0, l])
    fig.subplots_adjust(bottom=0.22)
    scatter = ax.scatter(samples[0][:, 0], samples[0][:, 1], alpha=alpha, s=3)
    slider = Slider(
        fig.add_axes([0.2, 0.1, 0.65, 0.03]),
        label="",
        valmin=0,
        valmax=len(samples),
        valstep=range(0, len(samples)),
    )
    slider.on_changed(lambda _: scatter.set_offsets(samples[slider.val]))
    plt.show()

class Config:
    def __init__(self, d):
        self.__dict__.update(d)