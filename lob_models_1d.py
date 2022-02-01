import torch
from torch import nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class ResBlock1d(nn.Module):
    def __init__(self, dim, kernel_size, activation=nn.LeakyReLU(0.01), dropout=0.0):
        super().__init__()
        dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.arch = nn.Sequential(
            nn.Conv1d(dim, dim // 2, kernel_size=1, padding="same"),
            nn.BatchNorm1d(dim // 2),
            dropout,
            activation,
            nn.Conv1d(dim // 2, dim // 2, kernel_size=kernel_size, padding="same"),
            nn.BatchNorm1d(dim // 2),
            dropout,
            activation,
            nn.Conv1d(dim // 2, dim, kernel_size=1, padding="same"),
            nn.BatchNorm1d(dim),
            dropout,
            activation,
        )

    def forward(self, x):
        return x + self.arch(x)


class CatModule(nn.Module):
    def __init__(self, models: list[nn.Module], dim):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.dim = dim

    def forward(self, x):
        return torch.cat([model(x) for model in self.models], dim=self.dim)


class Lob1dCNN(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        activation = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        conv1 = nn.Sequential(
            # nn.Conv2d(1, 16, kernel_size=(5,40),stride=(1,40),padding=(2,0)),
            # activation,
            # nn.BatchNorm2d(16),
            nn.Conv1d(40, 20, kernel_size=1, groups=20, padding="same"),
            nn.Conv1d(20, 10, kernel_size=1, groups=10, padding="same"),
        )
        conv2 = nn.Sequential(
            nn.Conv1d(40, 20, kernel_size=2, groups=20, padding="same"),
            nn.Conv1d(20, 10, kernel_size=2, groups=10, padding="same"),
        )
        conv3 = nn.Sequential(
            nn.Conv1d(40, 20, kernel_size=2, groups=1, padding="same"),
            nn.Conv1d(20, 10, kernel_size=2, groups=1, padding="same"),
        )

        self.arch = nn.Sequential(
            CatModule([conv1, conv2, conv3], dim=1),
            # [ Batch, C, Seq ]
            nn.Conv1d(30, 16, kernel_size=(3,), stride=2),
            nn.BatchNorm1d(16),
            activation,
            self.dropout,
            #
            # [ Batch, C, Seq // 2 ]
            nn.Conv1d(16, 32, kernel_size=(3,), stride=2),
            nn.BatchNorm1d(32),
            activation,
            #
            # ResBlock1d(32, (5,), dropout=dropout),
            #
            nn.Conv1d(32, 32, kernel_size=(3,), stride=2),
            nn.BatchNorm1d(32),
            activation,
            #
            nn.Conv1d(32, 64, kernel_size=(3,), stride=2),
            nn.BatchNorm1d(64),
            activation,
            # ResBlock1d(128, (2,), dropout=dropout),
            #
            nn.Conv1d(64, 64, kernel_size=(3,), padding="same"),
            nn.BatchNorm1d(64),
            activation,
            nn.MaxPool1d(2),
            #
            # nn.Conv1d(256, 256, kernel_size=(3,), stride=(2,)),
            # nn.BatchNorm1d(256),
            # activation,
            # # [ Batch, 32, Seq // 4 ]
            # nn.Conv1d(64,64, kernel_size=(3,)),
            # nn.BatchNorm1d(64),
            # activation,
            # # [ Batch, 32, Seq // 8 ]
            # nn.MaxPool1d(2),
            # [ Batch, 32 * Seq // 16 ] = [Batch, 2 * Seq]
            Rearrange("batch chan seq -> batch (chan seq)"),
            self.dropout,
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            self.dropout,
            activation,
            nn.Linear(64, 3),
        )

    def forward(self, x):
        return self.arch(x)


# test
if __name__ == "__main__":
    model = Lob1dCNN()
    model(torch.randn((2, 40, 100)))
