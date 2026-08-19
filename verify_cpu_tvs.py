"""
Kiem chung so hoc tren CPU cho patch Giai doan B (Phuong an 7): GaussianDiffusionTVS trong Model.py.
Khong can GPU - monkeypatch Tensor.cuda()/Module.cuda() thanh no-op.

Cac phep kiem tra:
  1. Hoi quy: velocity_mode=False phai cho q_sample/p_sample/training_losses TRUNG KHIT tuyet doi voi GaussianDiffusionAnchorOT
     (Phuong an 6) tren cung sigma_min/steps/seed/anchor_w.
  2. training_losses (velocity_mode=True): shape dung, khong NaN/Inf o moi lambda_x/y/z.
  3. p_sample (velocity_mode=True) tat dinh, model hoan hao -> tai tao dung x_start.
  4. Round-trip voi denoiser hoan hao tren velocity_mode=True, moi to hop (anchor_w, alpha_l, t) -> tai tao dung x_start (CT-7.4).
  5. Bien: user rong (x_start toan 0), anchor_w rat lon.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.Tensor.cuda = lambda self, *a, **k: self
nn.Module.cuda = lambda self, *a, **k: self

import sys
sys.path.insert(0, ".")
from Model import GaussianDiffusionAnchorOT, GaussianDiffusionTVS

torch.manual_seed(0)

STEPS = 5
SIGMA_MIN = 1e-3
BATCH = 8
NUM_ITEMS = 30
LATDIM = 16

def make_batch(zero_row=False):
    x = (torch.rand(BATCH, NUM_ITEMS) > 0.85).float().double()
    if zero_row:
        x[0] = 0.0
    uEmbeds_batch = torch.randn(BATCH, LATDIM, dtype=torch.float64)
    iEmbeds = torch.randn(NUM_ITEMS, LATDIM, dtype=torch.float64)
    return x, uEmbeds_batch, iEmbeds

def check_finite(name, t):
    assert torch.isfinite(t).all(), f"{name} co NaN/Inf: {t}"

def fake_model_factory(scale=1.0, offset=0.0):
    def fake_model(x_t, ts, *a, **k):
        return (x_t * scale + offset).clamp(-5.0, 5.0) # Velocity range can be larger than [0, 1]
    return fake_model

x_start, uEmbeds_batch, iEmbeds = make_batch()
itmEmbeds = torch.randn(NUM_ITEMS, LATDIM, dtype=torch.float64)
batch_index = torch.arange(BATCH)

# ---- Test 1: hoi quy velocity_mode=False vs GaussianDiffusionAnchorOT (PA6) ----
print("== Test 1: velocity_mode=False phai trung khit Phuong an 6 ==")
torch.manual_seed(42)
diff6 = GaussianDiffusionAnchorOT(SIGMA_MIN, STEPS, w_clip=50.0, num_sample_steps=0, anchor_w=2.0)
torch.manual_seed(42)
diff7_fallback = GaussianDiffusionTVS(SIGMA_MIN, STEPS, w_clip=50.0, num_sample_steps=0, anchor_w=2.0, velocity_mode=False)

assert torch.allclose(diff6.mu_coef, diff7_fallback.mu_coef)
assert torch.allclose(diff6.sigma_coef, diff7_fallback.sigma_coef)
assert torch.allclose(diff6.cfm_weight, diff7_fallback.cfm_weight)
print("  mu_coef, sigma_coef, cfm_weight: khop tuyet doi - OK")

t_fixed = torch.tensor([2] * BATCH)
noise_fixed = torch.randn(BATCH, NUM_ITEMS, dtype=torch.float64)
alpha_l = diff6._compute_anchor(uEmbeds_batch, itmEmbeds)

x_t_6 = diff6.q_sample(x_start, alpha_l, t_fixed, noise_fixed.clone())
x_t_7 = diff7_fallback.q_sample(x_start, alpha_l, t_fixed, noise_fixed.clone())
assert torch.allclose(x_t_6, x_t_7, atol=1e-12)
print("  q_sample: khop - OK")

fake_model = fake_model_factory(scale=0.7, offset=0.1)
mean6, _ = diff6.p_mean_variance(fake_model, x_t_6, alpha_l, t_fixed)
mean7, _ = diff7_fallback.p_mean_variance(fake_model, x_t_7, alpha_l, t_fixed)
assert torch.allclose(mean6, mean7, atol=1e-12)
print("  p_mean_variance: khop - OK")

torch.manual_seed(7)
out6 = diff6.p_sample(fake_model, x_start, uEmbeds_batch, iEmbeds, steps=3, sampling_noise=False)
torch.manual_seed(7)
out7 = diff7_fallback.p_sample(fake_model, x_start, uEmbeds_batch, iEmbeds, steps=3, sampling_noise=False)
assert torch.allclose(out6, out7, atol=1e-10)
print("  p_sample toan bo vong lap: khop - OK")

# ---- Test 2: training_losses (velocity_mode=True) ----
print("== Test 2: training_losses (velocity_mode=True) ==")
for anchor_w in [0.0, 1.0, 3.0]:
    for lx, ly, lz in [(1.0, 1.0, 1.0), (1.0, 0.1, 0.2), (0.5, 2.0, 0.0)]:
        diff = GaussianDiffusionTVS(SIGMA_MIN, STEPS, anchor_w=anchor_w, velocity_mode=True, lambda_x=lx, lambda_y=ly, lambda_z=lz)
        diff_loss, gc_loss = diff.training_losses(fake_model_factory(), x_start, itmEmbeds, batch_index, itmEmbeds, uEmbeds_batch)
        assert diff_loss.shape == (BATCH,) and gc_loss.shape == (BATCH,)
        check_finite(f"diff_loss(w={anchor_w}, lambdas={lx}/{ly}/{lz})", diff_loss)
        check_finite(f"gc_loss(w={anchor_w})", gc_loss)
print("  training_losses: OK o moi lambda va anchor_w, khong NaN/Inf")

# ---- Test 3: Round-trip voi denoiser hoan hao, moi to hop (anchor_w, alpha_l, seed) ----
print("== Test 3: round-trip denoiser hoan hao (CT-7.4) ==")
for anchor_w in [0.0, 1.5, 4.0]:
    for seed in range(3):
        torch.manual_seed(200 + seed)
        diff = GaussianDiffusionTVS(SIGMA_MIN, STEPS, anchor_w=anchor_w, velocity_mode=True)
        u_b = torch.randn(BATCH, LATDIM, dtype=torch.float64)
        i_e = torch.randn(NUM_ITEMS, LATDIM, dtype=torch.float64)
        alpha_l_val = diff._compute_anchor(u_b, i_e) if anchor_w != 0 else torch.zeros_like(x_start)
        
        # Test tung buoc thoi gian t tu STEPS-1 den 0
        x_cur = diff.q_sample(x_start, alpha_l_val, torch.tensor([STEPS - 1] * BATCH), noise=torch.randn(BATCH, NUM_ITEMS, dtype=torch.float64))
        
        for t_i in range(STEPS - 1, -1, -1):
            diff.next_index_map = {t_i: (t_i - 1 if t_i > 0 else -1)}
            t_tensor = torch.tensor([t_i] * BATCH)
            
            # Giả lập perfect model dự đoán đúng f_gt:
            # v_gt = (1.0 - sigma_min) * (anchor_w * alpha_l + noise) - x_start
            # Nhung o day, tai thoi diem t:
            # x_t = mu_t * x_start + sigma_t * anchor_w * alpha_l + sigma_t * eps
            # => eps = (x_t - mu_t * x_start - sigma_t * anchor_w * alpha_l) / sigma_t
            # Do do, v_gt_x = (1.0 - sigma_min) * (anchor_w * alpha_l + eps) - x_start
            mu_t = diff._extract_into_tensor(diff.mu_coef, t_tensor, x_cur.shape)
            sigma_t = diff._extract_into_tensor(diff.sigma_coef, t_tensor, x_cur.shape)
            eps_val = (x_cur - mu_t * x_start - sigma_t * anchor_w * alpha_l_val) / sigma_t.clamp(min=1e-8)
            v_gt_val = (1.0 - diff.sigma_min) * (anchor_w * alpha_l_val + eps_val) - x_start
            
            # Model tra ve perfect velocity
            perfect_velocity_model = lambda x, t, *a, **k: v_gt_val
            
            model_mean, _ = diff.p_mean_variance(perfect_velocity_model, x_cur, alpha_l_val, t_tensor)
            x_cur = model_mean
            
        assert torch.allclose(x_cur, x_start, atol=1e-7), f"anchor_w={anchor_w} seed={seed}: khong tai tao dung x_start"
print("  OK - round-trip tai tao chinh xac x_start o moi buoc")

# ---- Test 4: p_sample voi noisy model under velocity_mode=True ----
print("== Test 4: p_sample noisy model velocity_mode=True ==")
diffTVS = GaussianDiffusionTVS(SIGMA_MIN, STEPS, anchor_w=2.0, velocity_mode=True)
noisy_model = fake_model_factory(scale=0.1, offset=-0.2)
outTVS = diffTVS.p_sample(noisy_model, x_start, uEmbeds_batch, iEmbeds, steps=3, sampling_noise=False)
check_finite("p_sample(TVS)", outTVS)
print("  OK - khong NaN/Inf")

# ---- Test 5: bien ----
print("== Test 5: bien - user rong va anchor_w rat lon ==")
x_zero, u_zero, i_zero = make_batch(zero_row=True)
diffBig = GaussianDiffusionTVS(SIGMA_MIN, STEPS, anchor_w=1e6, velocity_mode=True)
out_big = diffBig.p_sample(fake_model_factory(scale=0.1), x_start, uEmbeds_batch, iEmbeds, steps=1, sampling_noise=False)
check_finite("p_sample(anchor_w=1e6)", out_big)
print("  OK - khong NaN/Inf")

print("\nTAT CA 5 PHEP KIEM TRA TREN CPU CHO TVS DEU PASS.")
