# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: ai4t
#     language: python
#     name: ai4t
# ---

# # LOB Dataset for Projects
#
# This jupyter notebook is used to download the FI-2010 [1] dataset for train and test a AI classifier on LOB data.
# The code is obtained from [2].
#
# ### Data:
# The FI-2010 is publicly avilable and interested readers can check out their paper [1]. The dataset can be downloaded from: https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649
#
# Otherwise, the notebook will download the data automatically or it can be obtained from:
#
# https://drive.google.com/drive/folders/1Xen3aRid9ZZhFqJRgEMyETNazk02cNmv?usp=sharing
#
# ### References:
# [1] Ntakaris A, Magris M, Kanniainen J, Gabbouj M, Iosifidis A. Benchmark dataset for mid‐price forecasting of limit order book data with machine learning methods. Journal of Forecasting. 2018 Dec;37(8):852-66. https://arxiv.org/abs/1705.03233
#
# [2] Zhang Z, Zohren S, Roberts S. DeepLOB: Deep convolutional neural networks for limit order books. IEEE Transactions on Signal Processing. 2019 Mar 25;67(11):3001-12. https://arxiv.org/abs/1808.03668
#
# ### This notebook runs on Pytorch 1.9.0.

# %load_ext autoreload
# %autoreload 2
# download the data
import os

if not os.path.isfile("data.zip"):
    # !wget https://raw.githubusercontent.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books/master/data/data.zip
    # !unzip -n data.zip
    print("data downloaded.")
else:
    print("data already existed.")

# !ls -la

# # Importing libraries
#
# ## My modules
#
# There are several modules outside of this notebook:
# - `loss.py` Contains `DynamicWeightCrossEntropy`, a loss function based on Cross Entropy, which estimates frequency for each label based on the input batches, and uses it to update an internal, running estimate of the label frequencies. This estimate is used as a weight to `torch.nn.functional.cross_entropy` in order to scale the loss value (and hence gradients w.r.t. model parameters) per-label.
# - `lob_dataset.py` Contains my implementation for the dataset of this task (`LobDataset`), and the dataset used by the DeepLOB paper (Zhang, Zohren, Roberts), as a check to see if the pre-processing steps match. My implementation optionally supports "window skipping", i.e. putting some evenly distributed space between the starting index of each input window; additionally, it is possible to use a `ShuffleDatasetIndices` callback which, at the end of each epoch, randomizes the starting point of each window. This allows for both the benefit of subsampling the dataset and having shorter epochs, without the drawback of unconditionally discarding training examples (which happens when using only window skipping).
# - `lob_models_1d.py` contains a purely sequential 1D convolutional neural network, to compare with the approaches of (Tsantekidis et al.) and (Zhang et al.).
# - `lob_models_2d.py` contains both the original DeepLOB model by the authors of the DeepLOB paper (called `TheirDeepLob`), and my reproduction of the model from (Tsantekidis et al.), called `Lob2dCNN`.
# - `lob_lightning_model.py` contains the `pytorch_lightning.LightningModule` which encapsulates training of each of the models described above. It takes care of optimization, computing metrics and logging them to Weights & Biases.

# +
# load packages
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
import pytorch_lightning as pl
from torchmetrics import functional as M

from loss import DynamicWeightCrossEntropy
from lob_datasets import LobDataset, TheirDataset
from lob_models_1d import Lob1dCNN, ResBlock1d
from lob_models_2d import TheirDeepLob, Lob2dCNN
from lightning_model import LobLightningModule

# -


# # Data preparation
#
# We used no auction dataset that is normalised by decimal precision approach in their work. The first seven days are training data and the last three days are testing data. A validation set (20%) from the training set is used to monitor the overfitting behaviours.
#
# The first 40 columns of the FI-2010 dataset are 10 levels ask and bid information for a limit order book and we only use these 40 features in our network. The last 5 columns of the FI-2010 dataset are the labels with different prediction horizons.

# +
# please change the data_path to your local path
# data_path = '/nfs/home/zihaoz/limit_order_book/data'
dec_data = np.loadtxt(
    "Train_Dst_NoAuction_DecPre_CF_7.txt"
)  # 80 training - 20 validation
dec_train = dec_data[:, : int(np.floor(dec_data.shape[1] * 0.8))]
dec_val = dec_data[:, int(np.floor(dec_data.shape[1] * 0.8)) :]

dec_test1 = np.loadtxt("Test_Dst_NoAuction_DecPre_CF_7.txt")
dec_test2 = np.loadtxt("Test_Dst_NoAuction_DecPre_CF_8.txt")
dec_test3 = np.loadtxt("Test_Dst_NoAuction_DecPre_CF_9.txt")
dec_test = np.hstack((dec_test1, dec_test2, dec_test3))

print(dec_train.shape, dec_val.shape, dec_test.shape)
# -

# all the data refer to 7 days, and the first 5 days are in the training set and validation
# and the last 2 days are inside the test set
x_training_data = dec_train.T[:, :40]
x_validation_data = dec_val.T[:, :40]
x_test_data = dec_test.T[:, :40]

print(x_training_data.shape, x_validation_data.shape, x_test_data.shape)

x_training_data[
    0
]  # 40 --> 10 levels and (ask-price, ask-volume, bid-price, bid-volume)

x_training_data[1]  # second time instant

TARGET_CLS_IDX = 4
y_training_data = dec_train[-5:].T[:, TARGET_CLS_IDX]
y_validation_data = dec_val[-5:].T[:, TARGET_CLS_IDX]
y_test_data = dec_test[-5:].T[:, TARGET_CLS_IDX]
print(y_training_data.shape)


# #### Dataset info:
#
# The 'x' is an 2d-array that contains, for each row a snapshot of the orderbook in the following structure:
# 'best-ask price', 'best-ask volume', 'best-bid price', 'best-bid volume', '2-lev ask price', '2-levl ask volume', '2-lev bid price', '2-lev bid volume', ....
#

# # My dataset instantiation
#
# I defined one dictionary of kwargs to pass to each dataset. Explanation of the parameters:
# - `window_len` is the length of input sequences for the model
# - `window_skip` controls the "window skipping" described previously. The starting index of each window is at a distance `window_skip` from the previous. Having `window_skip=2` means having half of the instances per-epoch. Using the `ShuffleDatasetIndices` callback affects this by adding a random number $r \in [0,\text{window_skip}$ used to offset the starting (and ending) point of each window.
# - `data_fmt` either `'2d'` or `'1d'`. Since I am using models which contain both 1D and 2D CNNs, I need to change the input shape according to the model at hand. This parameter controls the shape of input tensors, either `(batch, 1, height, width)` where `height` is sequence length, and `width` is the 40 orderbook levels in case of 2D CNNs, and `(batch, chan, sequence_len)` in the case of 1D CNNs, where `chan` equals 40.


# +
### commented because this code is quite expensive (and i don't use their datasets)
# their_dataset = TheirDataset(data=dec_train, k=4, num_classes=3, T=100)
# train_dataset = LobDataset(data=dec_train, k=4, num_classes=3, T=100)
# val_dataset = LobDataset(data=dec_val, k=4, num_classes=3, T=100)
# test_dataset = LobDataset(data=dec_test, k=4, num_classes=3, T=100)

dataset_kwargs = {
    "window_len": 100,
    "window_skip": 2,
    "data_fmt": "2d",  # or '2d' if using 2d cnns
}

train_dataset = dataset = LobDataset(x_training_data, y_training_data, **dataset_kwargs)
val_dataset = LobDataset(x_validation_data, y_validation_data, **dataset_kwargs)
test_dataset = LobDataset(x_test_data, y_test_data, **dataset_kwargs)
# print(dataset[0]['inputs'].shape, dataset[0]['labels'].shape)
dataset.inputs.shape, dataset.labels.shape
# -

elem = dataset[0]
seq, labels = elem.values()

# their_elem = their_dataset[0]
# their_seq, their_labels = their_elem
(
    seq.shape,
    labels.shape,
    # their_seq.shape, their_labels.shape
)


# +
# assert (seq == their_seq.to(seq.dtype)).all()
# assert (labels == their_labels.to(labels.dtype)).all()
# -
# ### The `ShuffleDatasetIndices` callback
#
# This is a pytorch lightning callback, and it is implemented by simply calling the `reset()` method of the `LobDataset`.


class ShuffleDatasetIndices(pl.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        trainer.train_dataloader.dataset.datasets.reset()

    def on_validation_epoch_start(self, trainer, pl_module):
        trainer.val_dataloaders[0].dataset.reset()


# # Training code
#
# Instantiation of the model, dataloaders, loggers and callbacks

# +
# try:
#     logger.experiment.finish()
# except:
#     pass
logger = pl.loggers.WandbLogger(project="ai4t-DeepLobster")
callbacks = [
    ShuffleDatasetIndices(),
    pl.callbacks.EarlyStopping(monitor="val/loss", patience=15),
]
# -


# #### Note on batch size
#
# Authors of (Tsantekidis et al.) use a very small batch size of 16, which also leads to slow training. They also don't report their learning rate, to the best of my knowledge. Hence I determined a LR of `1e-3` to reproduce their performance on the test set.
#
# I scale the batch size to $64 = 4*16$ and, following a common rule of thumb, set the learning rate to `4e-3`, scaling it by the same factor as the batch size.

# +
BATCH_SIZE = 64

# their_cnn = TheirDeepLob(dropout=0.5)

my_cnn = Lob2dCNN(dropout=0.1)

cnn = my_cnn
optimizer = torch.optim.Adam(cnn.parameters(), lr=4e-3)


model = LobLightningModule(
    model=cnn,
    opt=optimizer,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    test_dataset=test_dataset,
    batch_size=BATCH_SIZE,
    # loss_criterion=DynamicWeightCrossEntropy(n_classes=3, decay=0.9, minimum_weight=0.01),
)

train_dataloader = model.train_dataloader()
train_example_batch = next(iter(train_dataloader))
print({k: v.shape for k, v in train_example_batch.items()})

model_out = model(**train_example_batch)
print({k: v.shape for k, v in model_out.items()})
# -

from torchinfo import summary

shape = train_example_batch["inputs"].shape
print(shape)
summary(model.cuda(), shape, depth=5)

trainer = pl.Trainer(
    gpus=-1, benchmark=True, logger=logger, max_epochs=200, callbacks=callbacks
)

try:
    trainer.fit(model)
except KeyboardInterrupt:
    print("interrupting training....")
finally:
    test_result = trainer.test()
    print(test_result)
    logger.experiment.finish()

if False:
    logger.experiment.finish()

# model.loss.weight
