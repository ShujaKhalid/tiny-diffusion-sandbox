import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

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