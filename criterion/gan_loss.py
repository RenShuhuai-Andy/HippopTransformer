import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


def get_losses(d_out_real, d_out_fake, loss_type='JS'):
    """Get different adversarial losses according to given loss_type"""
    bce_loss = nn.BCEWithLogitsLoss()

    if loss_type == 'standard':  # the non-satuating GAN loss
        d_loss_real = bce_loss(d_out_real, torch.ones_like(d_out_real))
        d_loss_fake = bce_loss(d_out_fake, torch.zeros_like(d_out_fake))
        d_loss = d_loss_real + d_loss_fake

        g_loss = bce_loss(d_out_fake, torch.ones_like(d_out_fake))

    elif loss_type == 'JS':  # the vanilla GAN loss
        d_loss_real = bce_loss(d_out_real, torch.ones_like(d_out_real))
        d_loss_fake = bce_loss(d_out_fake, torch.zeros_like(d_out_fake))
        d_loss = d_loss_real + d_loss_fake

        g_loss = -d_loss_fake

    elif loss_type == 'KL':  # the GAN loss implicitly minimizing KL-divergence
        d_loss_real = bce_loss(d_out_real, torch.ones_like(d_out_real))
        d_loss_fake = bce_loss(d_out_fake, torch.zeros_like(d_out_fake))
        d_loss = d_loss_real + d_loss_fake

        g_loss = torch.mean(-d_out_fake)

    elif loss_type == 'hinge':  # the hinge loss
        d_loss_real = torch.mean(nn.ReLU(1.0 - d_out_real))
        d_loss_fake = torch.mean(nn.ReLU(1.0 + d_out_fake))
        d_loss = d_loss_real + d_loss_fake

        g_loss = -torch.mean(d_out_fake)

    elif 'wgan' in loss_type:  # 'wgan' or 'wgan-gp'
        d_loss_real = d_out_real.mean()
        d_loss_fake = d_out_fake.mean()
        d_loss = -d_loss_real + d_loss_fake
        g_loss = -d_loss_fake

    elif loss_type == 'tv':  # the total variation distance
        d_loss = torch.mean(nn.Tanh(d_out_fake) - nn.Tanh(d_out_real))
        g_loss = torch.mean(-nn.Tanh(d_out_fake))

    elif 'rsgan' in loss_type:  # 'rsgan' or 'rsgan-gp'
        d_loss = bce_loss(d_out_real - d_out_fake, torch.ones_like(d_out_real))
        g_loss = bce_loss(d_out_fake - d_out_real, torch.ones_like(d_out_fake))

    elif 'ppo' in loss_type:  # 'ppo' or 'ppo-gp'
        with torch.no_grad():
            W = d_out_fake.shape[0] * F.softmax(d_out_fake.data, dim=0)
        # loss_d_clas = (W*soft_logits).mean() - clas_logits.mean() + gp
        d_loss = torch.mean(W * d_out_fake - d_out_real)
        g_loss = -torch.mean(d_out_fake)

    else:
        raise NotImplementedError("Divergence '%s' is not implemented" % loss_type)

    return g_loss, d_loss

