from torch.utils.data import Dataset, DataLoader
from einops import rearrange
import torch
import numpy as np


def prepare_x(data):
    df1 = data[:40, :].T
    return np.array(df1)


def get_label(data):
    lob = data[-5:, :].T
    return lob


def data_classification(X, Y, T):
    [N, D] = X.shape
    df = np.array(X)

    dY = np.array(Y)

    dataY = dY[T - 1 : N]

    dataX = np.zeros((N - T + 1, T, D))
    for i in range(T, N + 1):
        dataX[i - T] = df[i - T : i, :]

    return dataX, dataY


def get_window_indices(n_elements: int, window_length: int, window_skip=1):
    """
    Constructs the list of window indices used to slice input and target tensors in
    the Dataset class.
    """
    return [range(i, i + window_length) for i in range(n_elements - window_length)][
        ::window_skip
    ]


class TheirDataset(Dataset):
    def __init__(self, data, k, num_classes, T):
        self.k = k
        self.num_classes = num_classes
        self.T = T
        x = prepare_x(data)
        y = get_label(data)
        x, y = data_classification(x, y, self.T)
        y = y[:, self.k] - 1
        self.length = len(x)

        x = torch.from_numpy(x)
        self.x = torch.unsqueeze(x, 1)
        self.y = torch.from_numpy(y)

    def __getitem__(self, index):
        """Generates samples of data"""
        return self.x[index], self.y[index]

    def __len__(self):
        """Denotes the total number of samples"""
        return self.length


class LobDataset(Dataset):
    def __init__(self, inputs, labels, window_len, data_fmt, window_skip=1):
        self.window_length = window_len
        self.window_skip = window_skip

        self.inputs = torch.from_numpy(inputs).float()
        self.labels = torch.from_numpy(labels).long() - 1

        self.data_fmt = data_fmt
        if data_fmt == "2d":
            self.inputs.unsqueeze_(0)
        elif data_fmt == "1d":
            self.inputs.transpose_(-1, -2)

        self.window_indices = torch.tensor(
            get_window_indices(
                n_elements=inputs.shape[0],
                window_length=self.window_length,
                window_skip=self.window_skip,
            ),
            dtype=torch.long,
        )[:-1]

        self._initial_window_indices = self.window_indices.clone()

    def reset(self):
        if self.window_skip == 1:
            return
        self.window_indices = self._initial_window_indices + torch.randint_like(
            self._initial_window_indices, low=0, high=self.window_skip - 1
        )

    def __getitem__(self, i):
        indices = self.window_indices[i]
        return {"inputs": self.inputs[:, indices], "labels": self.labels[indices[-1]]}

    def __len__(self):
        return len(self.window_indices)
