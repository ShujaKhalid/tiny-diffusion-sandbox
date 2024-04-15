import torch
import torch.nn as nn
from torchvision.ops import MLP


class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model_name = self.cfg.model

        # Initialize model here
        if self.model_name == "MLP":
            self.model = MLP(self.cfg)
        else:
            self.model = None

    def forward(self, x, t):
        return NotImplemented

class MLP(Model):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.input_dim = self.cfg.input_dim
        self.output_dim = self.cfg.output_dim
        self.hidden_dim = self.cfg.hidden_dim
        self.hidden_layers = self.cfg.hidden_layers

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            *(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.GELU()
            ) * self.hidden_layers
        )
        nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x, t):
        # Generate a tensor filled with the value of `t`
        # that is of the same size as the x tensor
        t_fill = torch.full((x.shape[0], 1), t)
        x = torch.cat((x, t_fill), axis=-1)
        return self.model(x)


