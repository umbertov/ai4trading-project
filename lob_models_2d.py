import torch
from torch import nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from einops import rearrange


class Lob2dCNN(nn.Module):
    def __init__(self, num_classes=3, dropout=0.0):
        super().__init__()
        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        activation = self.activation = nn.LeakyReLU(0.01)

        # convolution blocks
        self.conv1 = nn.Sequential(
            # 1)
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(4, 40)),
            activation,
            nn.BatchNorm2d(16),
            self.dropout,
        )
        self.conv2 = nn.Sequential(
            # 2)
            nn.Conv1d(in_channels=16, out_channels=16, kernel_size=4),
            nn.BatchNorm1d(16),
            activation,
            nn.MaxPool1d(2),
            # 3)
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3),
            nn.BatchNorm1d(32),
            activation,
            # 4)
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3),
            nn.BatchNorm1d(32),
            activation,
            nn.MaxPool1d(2),
        )
        self.fc = nn.Sequential(
            Rearrange("b c s -> b (c s)"),
            nn.Linear(672, 32),
            activation,
            nn.Linear(32, self.num_classes),
        )

    def forward(self, x):
        out = self.conv1(x).squeeze(-1)
        out = self.conv2(out)
        out = self.fc(out)
        return out


class TheirDeepLob(nn.Module):
    def __init__(self, num_classes=3, dropout=0.0):
        super().__init__()
        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # convolution blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=32, kernel_size=(1, 2), stride=(1, 2)
            ),
            nn.LeakyReLU(negative_slope=0.01),
            #             nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            self.dropout,
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=(1, 2), stride=(1, 2)
            ),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            self.dropout,
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 10)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            self.dropout,
        )

        # inception moduels
        self.inp1 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=(1, 1), padding="same"
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            self.dropout,
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=(3, 1), padding="same"
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )
        self.inp2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=(1, 1), padding="same"
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            self.dropout,
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=(5, 1), padding="same"
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )
        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=(1, 1), padding="same"
            ),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            self.dropout,
        )

        # lstm layers
        self.lstm = nn.LSTM(
            input_size=192, hidden_size=64, num_layers=1, batch_first=True
        )
        self.fc1 = nn.Linear(64, self.num_classes)

    def forward(self, x):
        # h0: (number of hidden layers, batch size, hidden size)
        h0 = torch.zeros(1, x.size(0), 64, device=x.device)
        c0 = torch.zeros(1, x.size(0), 64, device=x.device)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x_inp1 = self.inp1(x)
        x_inp2 = self.inp2(x)
        x_inp3 = self.inp3(x)

        x = torch.cat((x_inp1, x_inp2, x_inp3), dim=1)

        #         x = torch.transpose(x, 1, 2)
        x = x.permute(0, 2, 1, 3)
        x = torch.reshape(x, (-1, x.shape[1], x.shape[2]))

        x, _ = self.lstm(x, (h0, c0))
        x = x[:, -1, :]
        x = self.fc1(x)
        forecast_y = torch.softmax(x, dim=1)

        return forecast_y
