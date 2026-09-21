"""Kiem tra TVRec tren CPU: cong thuc TVS, backward va sampling.

Chay: python3 verify_cpu_tvs.py
Chi gia lap .cuda() trong pham vi kiem tra; training that van can CUDA.
"""
from unittest.mock import patch

import torch
from torch import nn

from Model import Denoise, GaussianDiffusionTVS


def main():
    torch.manual_seed(0)
    batch, items, latent = 8, 30, 16
    x = (torch.rand(batch, items) > 0.85).float()
    x[0] = 0  # User chua co tuong tac.
    users = torch.randn(batch, latent)
    item_embeds = torch.randn(items, latent)
    features = torch.randn(items, latent)
    indices = torch.arange(batch)

    # Cong thuc tai tao phai dung tai moi moc thoi gian, ke ca hai bien.
    for steps in (1, 5):
        for anchor_w in (0.0, 2.0, 4.0):
            diffusion = GaussianDiffusionTVS(1e-3, steps, anchor_w=anchor_w)
            anchor = diffusion._compute_anchor(users, item_embeds)
            for t in range(steps):
                ts = torch.full((batch,), t, dtype=torch.long)
                noise = torch.randn_like(x)
                noisy = diffusion.q_sample(x, anchor, ts, noise)
                velocity = (1 - diffusion.sigma_min) * (anchor_w * anchor + noise) - x
                sigma = diffusion._extract_into_tensor(diffusion.sigma_coef, ts, x.shape)
                reconstructed = (1 - diffusion.sigma_min) * noisy - sigma * velocity
                torch.testing.assert_close(reconstructed, x, rtol=0, atol=2e-6)

            # Oracle du doan van toc dung tren lich sampling that.
            def perfect_model(noisy, ts, *args):
                mu = diffusion._extract_into_tensor(diffusion.mu_coef, ts, x.shape)
                sigma = diffusion._extract_into_tensor(diffusion.sigma_coef, ts, x.shape)
                noise = (noisy - mu * x - sigma * anchor_w * anchor) / sigma
                return (1 - diffusion.sigma_min) * (anchor_w * anchor + noise) - x

            result = diffusion.p_sample(perfect_model, x, users, item_embeds, steps=steps)
            torch.testing.assert_close(result, x, rtol=0, atol=2e-6)
    print('OK: velocity reconstruction va perfect-model sampling (steps=1,5).')

    # Denoiser that: loss huu han, gradient den moi tham so, optimizer cap nhat.
    for anchor_w in (0.0, 2.0):
        for lx, ly, lz in ((1., 1., 1.), (1., 0., 0.), (0., 1., 0.), (0., 0., 1.)):
            diffusion = GaussianDiffusionTVS(
                1e-3, 5, anchor_w=anchor_w, lambda_x=lx, lambda_y=ly, lambda_z=lz,
            )
            model = Denoise([items, 12], [12, items], 10)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            before = model.in_layers[0].weight.detach().clone()
            diff_loss, gc_loss = diffusion.training_losses(
                model, x, item_embeds, indices, features, users,
            )
            assert diff_loss.shape == gc_loss.shape == (batch,)
            assert torch.isfinite(diff_loss).all() and torch.isfinite(gc_loss).all()
            (diff_loss.mean() + 0.1 * gc_loss.mean()).backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            optimizer.step()
            assert not torch.equal(before, model.in_layers[0].weight)
    print('OK: TVS loss, backward va Adam voi Denoise that, gom user rong.')

    with torch.no_grad():
        for anchor_w in (0.0, 2.0, 1e6):
            diffusion = GaussianDiffusionTVS(1e-3, 5, anchor_w=anchor_w)
            for sampling_steps in (0, 3, 5):
                torch.manual_seed(42)
                result = diffusion.p_sample(model, x, users, item_embeds, sampling_steps)
                torch.manual_seed(42)
                repeated = diffusion.p_sample(model, x, users, item_embeds, sampling_steps)
                assert result.shape == x.shape and torch.isfinite(result).all()
                torch.testing.assert_close(result, repeated, rtol=0, atol=0)
                assert torch.topk(result, k=1).indices.shape == (batch, 1)
    print('OK: sampling huu han, lap lai duoc voi cung seed, top-k rebuild dung shape.')
    print('Tat ca kiem tra CPU cho TVRec deu PASS.')


if __name__ == '__main__':
    with patch.object(torch.Tensor, 'cuda', lambda self, *a, **k: self), \
         patch.object(nn.Module, 'cuda', lambda self, *a, **k: self):
        main()
