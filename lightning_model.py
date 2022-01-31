import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
import pytorch_lightning as pl
from torchmetrics import functional as M


class LobLightningModule(pl.LightningModule):
    def __init__(
        self,
        model,
        opt,
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size,
        loss_criterion=nn.CrossEntropyLoss(),
    ):
        super().__init__()
        self.save_hyperparameters()
        self.opt = opt
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        self.model = model
        self.loss = loss_criterion

    def forward(self, inputs, **kwargs):
        out = {"model_out": self.model.forward(inputs)}
        if "labels" in kwargs:
            out["labels"] = kwargs["labels"]
        return out

    def step(self, inputs, labels, mode):
        model_out = self.model(inputs)  # :: [ Batch Classes ]
        loss = self.loss(model_out, labels.view(-1))
        self.log(f"{mode}/loss", loss)
        return {"model_out": model_out.detach(), "loss": loss, "labels": labels}

    def training_step(self, batch, batch_idx):
        return self.step(**batch, mode="train")

    def validation_step(self, batch, batch_idx):
        return self.step(**batch, mode="val")

    def test_step(self, batch, batch_idx):
        return self.step(**batch, mode="test")

    def epoch_end(self, step_outputs, mode):
        model_outs = torch.cat([x["model_out"] for x in step_outputs], dim=0)
        labels = torch.cat([x["labels"] for x in step_outputs], dim=0)

        accuracy = M.accuracy(model_outs, labels, average="macro", num_classes=3)
        f1_macro = M.f1(model_outs, labels, average="macro", num_classes=3)
        f1_micro = M.f1(model_outs, labels, average="micro", num_classes=3)
        self.logger.experiment.log(
            {
                f"{mode}/accuracy": accuracy,
                f"{mode}/f1_macro": f1_macro,
                f"{mode}/f1_micro": f1_micro,
            }
        )

    def training_epoch_end(self, step_outputs):
        return self.epoch_end(step_outputs, mode="train")

    def validation_epoch_end(self, step_outputs):
        return self.epoch_end(step_outputs, mode="val")

    def test_epoch_end(self, step_outputs):
        return self.epoch_end(step_outputs, mode="test")

    def configure_optimizers(self, *args, **kwargs):
        return self.opt

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=4,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=4,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=4,
        )
