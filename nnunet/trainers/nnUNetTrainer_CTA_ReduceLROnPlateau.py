from __future__ import annotations

import numpy as np
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_CTA_ReduceLROnPlateau(nnUNetTrainer):
    """
    nnUNet trainer variant for CTA artery segmentation.

    - Optimizer: SGD with Nesterov momentum (mu=0.99)
    - LR schedule: ReduceLROnPlateau on validation loss
    - Initial LR: 0.01 (default nnUNetv2)
    """

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.network.parameters(),
            self.initial_lr,
            weight_decay=self.weight_decay,
            momentum=0.99,
            nesterov=True,
        )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=30,
            threshold=1e-4,
            min_lr=1e-6,
            verbose=True,
        )
        return optimizer, lr_scheduler

    def on_train_epoch_start(self):
        self.network.train()
        self.print_to_log_file("")
        self.print_to_log_file(f"Epoch {self.current_epoch}")
        self.print_to_log_file(
            f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}"
        )
        self.logger.log("lrs", self.optimizer.param_groups[0]["lr"], self.current_epoch)

    def on_epoch_end(self):
        if self.lr_scheduler is not None:
            val_losses = self.logger.my_fantastic_logging.get("val_losses", [])
            if val_losses:
                self.lr_scheduler.step(float(val_losses[-1]))
        super().on_epoch_end()
