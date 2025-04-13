"""
CycleGAN implementation using PyTorch Lightning.

This class encapsulates the full training logic for CycleGAN, including the architecture,
loss computation, training steps, and learning rate scheduling.

Key components:
- Two Generators: G_PS (Photo → Style), G_SP (Style → Photo)
- Two Discriminators: D_P (Photo domain), D_S (Style domain)
- Replay buffers to stabilize discriminator training
"""

import pytorch_lightning as L
import torch
import torch.nn.functional as F
from torch import nn

from .buffer import ReplayBuffer
from .discriminator import Discriminator

from .generator import Generator
from .utils import show_img


class CycleGAN(L.LightningModule):
    """
    CycleGAN LightningModule.

    Args:
        name (str): Style name or dataset identifier.
        num_resblocks (int): Number of residual blocks in generator.
        hidden_channels (int): Base number of channels.
        optimizer (torch.optim.Optimizer): Optimizer class.
        lr (float): Learning rate.
        betas (tuple): Betas for Adam optimizer.
        lambda_idt (float): Identity loss weight.
        lambda_cycle (tuple): Cycle consistency loss weights for both directions.
        buffer_max_size (int): Replay buffer size.
        num_epochs (int): Total training epochs.
        decay_epochs (int): Epoch to start learning rate decay.
    """

    def __init__(
        self,
        name,
        num_resblocks,
        hidden_channels,
        optimizer,
        lr,
        betas,
        lambda_idt,
        lambda_cycle,
        buffer_max_size,
        num_epochs,
        decay_epochs,
    ):
        super().__init__()

        self.optimizer = optimizer
        self.save_hyperparameters(ignore=["optimizer"])
        self.automatic_optimization = False

        self.G_PS = Generator(name, hidden_channels, num_resblocks).create()
        self.G_SP = Generator(name, hidden_channels, num_resblocks).create()

        self.D_P = Discriminator(hidden_channels)
        self.D_S = Discriminator(hidden_channels)

        self.fake_P_buffer = ReplayBuffer(buffer_max_size)
        self.fake_S_buffer = ReplayBuffer(buffer_max_size)

    def forward(self, img):
        """Forward pass: Photo to Style."""

        return self.G_PS(img)

    def init_weights(self):
        """Initialize model weights using normal distribution."""

        def init_fn(m):
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.InstanceNorm2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        for net in [self.G_PS, self.G_SP, self.D_S, self.D_P]:
            net.apply(init_fn)

    def setup(self, stage):
        if stage == "fit":
            self.init_weights()

    def get_lr_scheduler(self, optimizer):
        """Custom learning rate scheduler with linear decay."""

        def lr_lambda(epoch):
            len_decay_phase = self.hparams.num_epochs - self.hparams.decay_epochs + 1.0
            curr_decay_step = max(0, epoch - self.hparams.decay_epochs + 1.0)
            val = 1.0 - curr_decay_step / len_decay_phase

            return max(0.0, val)

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    def configure_optimizers(self):
        """Define optimizers and learning rate schedulers."""

        optimizer_config = {"lr": self.hparams.lr, "betas": self.hparams.betas}

        optimizer_G = self.optimizer(
            list(self.G_PS.parameters()) + list(self.G_SP.parameters()),
            **optimizer_config,
        )
        optimizer_D = self.optimizer(
            list(self.D_S.parameters()) + list(self.D_P.parameters()),
            **optimizer_config,
        )

        optimizers = [optimizer_G, optimizer_D]
        schedulers = [self.get_lr_scheduler(opt) for opt in optimizers]

        return optimizers, schedulers

    def training_step(self, batch, batch_idx):
        """Main training logic with manual optimization steps."""

        self.real_S = batch["style"]
        self.real_P = batch["photo"]
        opt_gen, opt_disc = self.optimizers()

        # Generator forward pass

        self.fake_S = self.G_PS(self.real_P)
        self.fake_P = self.G_SP(self.real_S)

        self.idt_S = self.G_PS(self.real_S)
        self.idt_P = self.G_SP(self.real_P)

        self.recon_S = self.G_PS(self.fake_P)
        self.recon_P = self.G_SP(self.fake_S)

        # train generators

        self.toggle_optimizer(opt_gen)
        gen_loss = self.get_gen_loss()
        opt_gen.zero_grad()
        self.manual_backward(gen_loss)
        opt_gen.step()
        self.untoggle_optimizer(opt_gen)

        # Train discriminators

        self.toggle_optimizer(opt_disc)
        disc_loss_S = self.get_disc_loss_S()
        disc_loss_P = self.get_disc_loss_P()
        opt_disc.zero_grad()
        self.manual_backward(disc_loss_S)
        self.manual_backward(disc_loss_P)
        opt_disc.step()
        self.untoggle_optimizer(opt_disc)

        # record training losses

        metrics = {
            "gen_loss": gen_loss,
            "disc_loss_S": disc_loss_S,
            "disc_loss_P": disc_loss_P,
        }
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)

    def get_cycle_loss(self, real, recon, lambda_cycle):
        """Cycle consistency loss (L1)."""
        cycle_loss = F.l1_loss(recon, real)
        return lambda_cycle * cycle_loss

    def get_adv_loss(self, fake, disc):
        """Adversarial loss (LSGAN)."""
        fake_hat = disc(fake)
        real_labels = torch.ones_like(fake_hat)
        adv_loss = F.mse_loss(fake_hat, real_labels)
        return adv_loss

    def get_idt_loss(self, real, idt, lambda_cycle):
        """Identity loss (L1)."""
        idt_loss = F.l1_loss(idt, real)
        return self.hparams.lambda_idt * lambda_cycle * idt_loss

    def get_gen_loss(self):
        """Full generator loss: adversarial + identity + cycle."""

        # calculate adversarial loss

        adv_loss_PS = self.get_adv_loss(self.fake_S, self.D_S)
        adv_loss_SP = self.get_adv_loss(self.fake_P, self.D_P)
        total_adv_loss = adv_loss_PS + adv_loss_SP

        # calculate identity loss

        lambda_cycle = self.hparams.lambda_cycle
        idt_loss_SS = self.get_idt_loss(self.real_S, self.idt_S, lambda_cycle[0])
        idt_loss_PP = self.get_idt_loss(self.real_P, self.idt_P, lambda_cycle[1])
        total_idt_loss = idt_loss_SS + idt_loss_PP

        # calculate cycle loss

        cycle_loss_SPS = self.get_cycle_loss(self.real_S, self.recon_S, lambda_cycle[0])
        cycle_loss_PSP = self.get_cycle_loss(self.real_P, self.recon_P, lambda_cycle[1])
        total_cycle_loss = cycle_loss_SPS + cycle_loss_PSP

        # combine losses

        gen_loss = total_adv_loss + total_idt_loss + total_cycle_loss
        return gen_loss

    def get_disc_loss(self, real, fake, disc):
        """Discriminator loss with real and fake images."""
        real_hat = disc(real)
        real_labels = torch.ones_like(real_hat)
        real_loss = F.mse_loss(real_hat, real_labels)

        fake_hat = disc(fake.detach())
        fake_labels = torch.zeros_like(fake_hat)
        fake_loss = F.mse_loss(fake_hat, fake_labels)

        disc_loss = (fake_loss + real_loss) * 0.5
        return disc_loss

    def get_disc_loss_S(self):
        fake_S = self.fake_S_buffer(self.fake_S)
        return self.get_disc_loss(self.real_S, fake_S, self.D_S)

    def get_disc_loss_P(self):
        fake_P = self.fake_P_buffer(self.fake_P)
        return self.get_disc_loss(self.real_P, fake_P, self.D_P)

    def validation_step(self, batch, batch_idx):
        self.display_results(batch, batch_idx, "validate")

    def display_results(self, batch, batch_idx, stage):
        real_P = batch
        fake_S = self(real_P)

        if stage == "validate":
            title = f"Epoch {self.current_epoch + 1}: Photo-to-Style Translation"
        else:
            title = f"Sample {batch_idx + 1}: Photo-to-Style Translation"
        show_img(
            torch.cat([real_P, fake_S], dim=0),
            nrow=len(real_P),
            title=title,
        )

    def on_train_epoch_start(self):
        curr_lr = self.lr_schedulers()[0].get_last_lr()[0]
        self.log("lr", curr_lr, on_step=False, on_epoch=True, prog_bar=True)
        # logger.report_scalar("Learning Rate", "train", value=curr_lr, iteration=self.current_epoch)

    def on_train_epoch_end(self):
        avg_gen_loss = self.trainer.callback_metrics["gen_loss"].item()
        avg_disc_loss_S = self.trainer.callback_metrics["disc_loss_S"].item()
        avg_disc_loss_P = self.trainer.callback_metrics["disc_loss_P"].item()

        for sch in self.lr_schedulers():
            sch.step()
        print(
            f"Epoch {self.current_epoch + 1}",
            f"gen_loss: {avg_gen_loss:.5f}",
            f"disc_loss_S: {avg_disc_loss_S:.5f}",
            f"disc_loss_P: {avg_disc_loss_P:.5f}",
            sep=" - ",
        )

    def on_train_end(self):
        print("Training ended.")

    def on_predict_epoch_end(self):
        predictions = self.trainer.predict_loop.predictions
        num_batches = len(predictions)
        batch_size = predictions[0].shape[0]
        last_batch_diff = batch_size - predictions[-1].shape[0]
        print(
            f"Number of images generated: {num_batches * batch_size - last_batch_diff}."
        )
