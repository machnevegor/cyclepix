import torch
from tqdm import tqdm
import torch.nn as nn
from torchvision.utils import save_image


def train(
        d_real, d_style, g_real, g_style,
        data_loader, d_optim, g_optim,
        d_scaler, g_scaler,
        lambda_cycle=10,
        lambda_identity=5
):
    """
    Train function for CycleGAN without Replay Buffer.

    Args:
        d_real (nn.Module): Discriminator for real images.
        d_style (nn.Module): Discriminator for style images.
        g_real (nn.Module): Generator for style-to-real transformation.
        g_style (nn.Module): Generator for real-to-style transformation.
        data_loader (DataLoader): Dataloader providing batches of (real, style) images.
        d_optim (Optimizer): Optimizer for discriminators.
        g_optim (Optimizer): Optimizer for generators.
        d_scaler (GradScaler): Mixed precision scaler for discriminators.
        g_scaler (GradScaler): Mixed precision scaler for generators.
        lambda_cycle (float): Weight for cycle consistency loss.
        lambda_identity (float): Weight for identity loss.

    Returns:
        tuple: Lists of discriminator and generator losses over the epoch.
    """
    loop = tqdm(data_loader, leave=True)

    mse = nn.MSELoss()
    l1 = nn.L1Loss()

    d_loss_lst = []
    g_loss_lst = []

    for i, (real_img, style_img) in enumerate(loop):
        real_img = real_img.cuda()
        style_img = style_img.cuda()

        # ------------------------------------
        # 1) Train Discriminators
        # ------------------------------------
        with torch.cuda.amp.autocast():
            fake_style = g_style(real_img).detach()  # Detach to avoid updating generator
            fake_real = g_real(style_img).detach()

            # Discriminator loss for style images
            d_style_real = d_style(style_img)
            d_style_fake = d_style(fake_style)
            d_style_real_loss = mse(d_style_real, torch.ones_like(d_style_real))
            d_style_fake_loss = mse(d_style_fake, torch.zeros_like(d_style_fake))
            d_style_loss = (d_style_real_loss + d_style_fake_loss) * 0.5

            # Discriminator loss for real images
            d_real_real = d_real(real_img)
            d_real_fake = d_real(fake_real)
            d_real_real_loss = mse(d_real_real, torch.ones_like(d_real_real))
            d_real_fake_loss = mse(d_real_fake, torch.zeros_like(d_real_fake))
            d_real_loss = (d_real_real_loss + d_real_fake_loss) * 0.5

            d_loss = d_style_loss + d_real_loss

        d_optim.zero_grad()
        d_scaler.scale(d_loss).backward()
        d_scaler.step(d_optim)
        d_scaler.update()

        # ------------------------------------
        # 2) Train Generators
        # ------------------------------------
        with torch.cuda.amp.autocast():
            fake_style = g_style(real_img)
            fake_real = g_real(style_img)

            # Adversarial loss
            d_style_fake = d_style(fake_style)
            d_real_fake = d_real(fake_real)
            g_style_loss = mse(d_style_fake, torch.ones_like(d_style_fake))
            g_real_loss = mse(d_real_fake, torch.ones_like(d_real_fake))

            # Cycle consistency loss
            cycle_real = g_real(fake_style)
            cycle_style = g_style(fake_real)
            cycle_real_loss = l1(cycle_real, real_img)
            cycle_style_loss = l1(cycle_style, style_img)

            # Identity loss
            identity_real = g_real(real_img)
            identity_style = g_style(style_img)
            identity_real_loss = l1(identity_real, real_img)
            identity_style_loss = l1(identity_style, style_img)

            # Total generator loss
            g_loss = (
                    g_style_loss + g_real_loss
                    + cycle_real_loss * lambda_cycle
                    + cycle_style_loss * lambda_cycle
                    + identity_real_loss * lambda_identity
                    + identity_style_loss * lambda_identity
            )

        g_optim.zero_grad()
        g_scaler.scale(g_loss).backward()
        g_scaler.step(g_optim)
        g_scaler.update()

        # Logging losses
        loop.set_postfix({
            "d_loss": d_loss.item(),
            "g_loss": g_loss.item()
        })

        d_loss_lst.append(d_loss.item())
        g_loss_lst.append(g_loss.item())

        # Save images every 300 iterations
        if i % 300 == 0:
            save_image(fake_style * 0.5 + 0.5, f"saved_images/style_{i}.png")
            save_image(fake_real * 0.5 + 0.5, f"saved_images/real_{i}.png")

    return d_loss_lst, g_loss_lst
