import sys
from typing import List, Optional

import lightning as L
import numpy as np
import torch
import torch.nn as nn

from torchmetrics import AUROC
from h5dataset import outcome_labels


class MALBECAUROC:
    """Binary AUROC class"""
    def __init__(
        self,
        label_names: List[str],
        compute_on_cpu: bool = False,
        sync_on_compute: bool = True,  # Required for DDP
    ):
        self.label_names = label_names
        self.metrics = {}

        # Create a binary AUROC metric for each label
        for label in label_names:
            self.metrics[label] = AUROC(
                task="binary",  # Always use binary task for per-label computation
                num_classes=1,  # Treat each label as a binary classification
                compute_on_cpu=compute_on_cpu,
                sync_on_compute=sync_on_compute,
            )

        # Precompute indices for critical labels
        self.stemi_idx = label_names.index("stemi_hd_v2") if "stemi_hd_v2" in label_names else None
        self.nstemi_idx = label_names.index("nstemi_hd_v2") if "nstemi_hd_v2" in label_names else None
        self.nstemi_type1_idx = label_names.index("nstemi_type1_hd_v2") if "nstemi_type1_hd_v2" in label_names else None

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # Ensure preds and target are 2D tensors [batch_size, num_labels]
        assert preds.ndim == 2 and target.ndim == 2, "preds and target must be 2D"

        # For stemi_hd_v2: Use all samples (no filtering)
        if "stemi_hd_v2" in self.metrics:
            self.metrics["stemi_hd_v2"].update(
                preds[:, self.stemi_idx],  # 1D tensor [batch_size]
                target[:, self.stemi_idx].int()  # 1D tensor [batch_size]
            )

        # For non-stemi outcomes: Filter out stemi_hd_v2 == 1
        if self.stemi_idx is not None:
            mask = target[:, self.stemi_idx] == 0  # Mask where stemi_hd_v2 is 0
            preds_filtered = preds[mask]
            target_filtered = target[mask]
        else:
            preds_filtered = preds
            target_filtered = target

        # Process nstemi_hd_v2: max(stemi_prob, nstemi_prob) BEFORE filtering
        if "nstemi_hd_v2" in self.metrics and self.stemi_idx is not None and self.nstemi_idx is not None:
            nstemi_probs = torch.max(preds[:, self.stemi_idx], preds[:, self.nstemi_idx])
            self.metrics["nstemi_hd_v2"].update(
                nstemi_probs[mask],  # 1D tensor [filtered_batch_size]
                target_filtered[:, self.nstemi_idx].int()
            )

        # Process nstemi_type1_hd_v2: max(stemi_prob, nstemi_type1_prob) BEFORE filtering
        if "nstemi_type1_hd_v2" in self.metrics and self.stemi_idx is not None and self.nstemi_type1_idx is not None:
            nstemi_type1_probs = torch.max(preds[:, self.stemi_idx], preds[:, self.nstemi_type1_idx])
            self.metrics["nstemi_type1_hd_v2"].update(
                nstemi_type1_probs[mask],  # 1D tensor [filtered_batch_size]
                target_filtered[:, self.nstemi_type1_idx].int()
            )

        # Process other labels (filtered but no adjustment)
        for label in self.label_names:
            if label in ["stemi_hd_v2", "nstemi_hd_v2", "nstemi_type1_hd_v2"]:
                continue  # Already handled
            idx = self.label_names.index(label)
            self.metrics[label].update(
                preds_filtered[:, idx],  # 1D tensor [filtered_batch_size]
                target_filtered[:, idx].int()
            )

    def compute(self):
        return {label: metric.compute() for label, metric in self.metrics.items()}

    def reset(self):
        for metric in self.metrics.values():
            metric.reset()
  

def _padding(downsample: int, kernel_size: int) -> int:
    """Calculate padding for convolution."""
    padding = max(0, int(np.floor((kernel_size - downsample + 1) / 2)))
    return padding


def _downsample(n_samples_in: int, n_samples_out: int) -> int:
    """Calculate downsample rate."""
    downsample = int(n_samples_in // n_samples_out)
    if downsample < 1:
        raise ValueError("Number of samples should always decrease")
    if n_samples_in % n_samples_out != 0:
        raise ValueError(
            "Number of samples for two consecutive blocks "
            "should always decrease by an integer factor."
        )
    return downsample


class ResBlock1d(nn.Module):
    """Residual network unit for unidimensional signals."""

    def __init__(
        self,
        n_filters_in: int,
        n_filters_out: int,
        downsample: int,
        kernel_size: int,
        dropout_rate: float
    ):
        """Initialize residual block layers."""
        if kernel_size % 2 == 0:
            raise ValueError(
                "The current implementation only support odd values for `kernel_size`."
            )
        super(ResBlock1d, self).__init__()
        padding = _padding(1, kernel_size)
        self.conv1 = nn.Conv1d(
            n_filters_in, n_filters_out, kernel_size, padding=padding, bias=False
        )
        self.bn1 = nn.BatchNorm1d(n_filters_out)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        padding = _padding(downsample, kernel_size)
        self.conv2 = nn.Conv1d(
            n_filters_out,
            n_filters_out,
            kernel_size,
            stride=downsample,
            padding=padding,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(n_filters_out)
        self.dropout2 = nn.Dropout(dropout_rate)

        skip_connection_layers = []
        if downsample > 1:
            maxpool = nn.MaxPool1d(downsample, stride=downsample)
            skip_connection_layers += [maxpool]
        if n_filters_in != n_filters_out:
            conv1x1 = nn.Conv1d(n_filters_in, n_filters_out, 1, bias=False)
            skip_connection_layers += [conv1x1]
        self.skip_connection = nn.Sequential(*skip_connection_layers) if skip_connection_layers else None

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple:
        """Process input through residual unit."""
        if self.skip_connection is not None:
            y = self.skip_connection(y)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x += y
        y = x
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        return x, y


class ResNet1d(nn.Module):
    """Residual network for unidimensional signals."""

    def __init__(
        self,
        input_dim: tuple,
        blocks_dim: list,
        n_classes: int,
        kernel_size: int = 17,
        dropout_rate: float = 0.8
    ):
        """Initialize ResNet1d architecture."""
        super(ResNet1d, self).__init__()
        n_filters_in, n_filters_out = input_dim[0], blocks_dim[0][0]
        n_samples_in, n_samples_out = input_dim[1], blocks_dim[0][1]
        downsample = _downsample(n_samples_in, n_samples_out)
        padding = _padding(downsample, kernel_size)
        self.conv1 = nn.Conv1d(
            n_filters_in,
            n_filters_out,
            kernel_size,
            bias=False,
            stride=downsample,
            padding=padding,
        )
        self.bn1 = nn.BatchNorm1d(n_filters_out)

        res_blocks = []
        for i, (n_filters, n_samples) in enumerate(blocks_dim):
            n_filters_in, n_filters_out = n_filters_out, n_filters
            n_samples_in, n_samples_out = n_samples_out, n_samples
            downsample = _downsample(n_samples_in, n_samples_out)
            resblk1d = ResBlock1d(
                n_filters_in, n_filters_out, downsample, kernel_size, dropout_rate
            )
            self.add_module(f"resblock1d_{i}", resblk1d)
            res_blocks += [resblk1d]
        self.res_blocks = nn.Sequential(*res_blocks)

        n_filters_last, n_samples_last = blocks_dim[-1]
        last_layer_dim = n_filters_last * n_samples_last
        self.lin = nn.Linear(last_layer_dim, n_classes)
        self.n_blk = torch.tensor(len(blocks_dim), dtype=torch.int64)
        
        self.initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform forward pass through ResNet1d."""
        x = self.conv1(x.float())
        x = self.bn1(x)
        y = x
        for blk in self.res_blocks:
            x, y = blk(x, y)
        x = x.view(x.size(0), -1)
        x = self.lin(x)
        return x

    def initialize_weights(self):
        """Initialize weights of Conv1d and Linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)


def load_weights(path: str) -> dict:
    """Load and process model weights."""
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    checkpoint = checkpoint["model"]
    checkpoint.pop("lin.weight", None)
    checkpoint.pop("lin.bias", None)
    new_state_dict = {}
    for key, value in checkpoint.items():
        new_key = key.replace('resblock1d_', 'res_blocks.') if 'resblock1d_' in key else key
        new_state_dict[new_key] = value
    return new_state_dict


class LitResNet(L.LightningModule):
    """LightningModule for Ribeiro ResNet model."""

    def __init__(self, hparams: dict = None):
        """Initialize model parameters."""
        super(LitResNet, self).__init__()
        self.save_hyperparameters(hparams)
        self.model = ResNet1d(
            input_dim=(self.hparams["n_leads"], self.hparams["seq_length"]),
            blocks_dim=list(zip(self.hparams["net_filter_size"], self.hparams["net_seq_length"])),
            n_classes=self.hparams["n_classes"],
            kernel_size=self.hparams["kernel_size"],
            dropout_rate=self.hparams["dropout_rate"],
        )
        self.conv_fc_params = [
            param for name, param in self.named_parameters()
            if ('conv' in name.lower() or 'fc' in name.lower() or 'lin' in name.lower()) and param.requires_grad
        ]
        self.reinit()

    def reinit(self):
        """Reset metrics and loss function."""
        self.train_auroc = MALBECAUROC(
            label_names=outcome_labels,
            compute_on_cpu=False,
            sync_on_compute=False,              
        )

        self.val_auroc = MALBECAUROC(
            label_names=outcome_labels,
            compute_on_cpu=False,
            sync_on_compute=False,
        )
        self.outcome_labels = outcome_labels
        self.loss_fn = nn.BCEWithLogitsLoss(weight=torch.tensor(list(self.hparams["weights"].values()))
                                            / self.hparams['n_classes'])
        self.grad_history = []
        self.gradient_clipping = self.hparams["gradient_clipping"]
        
        if self.hparams["freeze"] > 0:
            n_frozen_blocks = self.hparams["freeze"]
            for i in range(n_frozen_blocks):
                for param in self.model.res_blocks[i].parameters():
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through model."""
        return self.model(x)

    def configure_optimizers(self) -> dict:
        """Set up optimizer and scheduler."""
        conv_fc_params = [
            param for name, param in self.named_parameters()
            if ('conv' in name.lower() or 'fc' in name.lower() or 'lin' in name.lower()) and param.requires_grad
        ]
        other_params = [
            param for name, param in self.named_parameters()
            if ('conv' not in name.lower() and 'fc' not in name.lower() and 'lin' not in name.lower()) and param.requires_grad
        ]
                
        optimizer = AdamP(
            params=
            [
                {'params': conv_fc_params,'weight_decay': self.hparams["l2_regularization"]},
                {'params': other_params,'weight_decay': 0}
            ],
            lr=self.hparams["lr_max"],
            betas=(0.90, 0.999)
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            min_lr=self.hparams["lr_min"],
            patience=self.hparams["lr_patience"],
            factor=self.hparams["lr_factor"]
        )
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def training_step(self, train_batch: dict) -> torch.Tensor:
        """Train with regularization."""
        x, y = train_batch["ecg"], train_batch["target"]
        y_hat = self.forward(x.float())
        base_loss = self.loss_fn(y_hat, y.float())
        loss = base_loss 
        
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=False,
        )
        self.train_auroc.update(preds=y_hat, target=y.long())
        return loss

    def on_train_epoch_end(self):
        """Log training AUROC."""
        auroc_values = self.train_auroc.compute()
        for i,label in enumerate(auroc_values):
            self.log(f"train_auroc_class_{label}", auroc_values[label], sync_dist=True)
        self.train_auroc.reset()

    def validation_step(self, val_batch: dict) -> torch.Tensor:
        """Validate without regularization."""
        x, y = val_batch["ecg"], val_batch["target"]
        y_hat = self.forward(x.float())
        base_loss = self.loss_fn(y_hat, y.float())
        loss = base_loss
        
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=False,
        )
        self.val_auroc.update(preds=y_hat, target=y.long())
        return loss

    def on_validation_epoch_end(self):
        """Log validation AUROC."""
        auroc_values = self.val_auroc.compute()
        for i,label in enumerate(auroc_values):
            self.log(f"val_auroc_class_{label}", auroc_values[label], sync_dist=True)
        self.val_auroc.reset()

    def compute_grad_norm(self, model):
        """Calculate gradient norm."""
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm

    def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
        """Apply adaptive gradient clipping."""
        grad_norm = self.compute_grad_norm(self.model)
        self.grad_history.append(grad_norm)
        self.grad_history = self.grad_history[-1000:]
        clip_value = np.percentile(self.grad_history, self.gradient_clipping)
        self.clip_gradients(optimizer, gradient_clip_val=clip_value,
                            gradient_clip_algorithm=gradient_clip_algorithm)