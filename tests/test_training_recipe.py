"""Tests for the paper's training-recipe components (Table 2): LR warmup +
cosine annealing, EMA of encoder weights, and encoder dropout / stochastic
depth. These were flagged missing in VERIFICATION_REPORT_round4.md and added
in this round; these tests pin the expected behaviour down.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from edcl.schedule import build_warmup_cosine_scheduler, warmup_cosine_lambda
from edcl.ema import EMA
from edcl.backbone import EquivariantEncoder


def test_warmup_cosine_lambda_shape():
    warmup, total = 10, 100
    # ramps up linearly during warmup
    assert warmup_cosine_lambda(0, warmup, total) == 1 / warmup
    assert math.isclose(warmup_cosine_lambda(warmup - 1, warmup, total), 1.0, rel_tol=1e-6)
    # peaks at 1.0 right after warmup, decays to min_lr_ratio at total_steps
    peak = warmup_cosine_lambda(warmup, warmup, total, min_lr_ratio=0.0)
    end = warmup_cosine_lambda(total, warmup, total, min_lr_ratio=0.05)
    assert peak > 0.9
    assert math.isclose(end, 0.05, abs_tol=1e-6)
    # monotonically non-increasing after warmup
    vals = [warmup_cosine_lambda(s, warmup, total) for s in range(warmup, total + 1)]
    assert all(a >= b - 1e-9 for a, b in zip(vals, vals[1:]))


def test_scheduler_matches_lambda_on_optimizer_lr():
    model = nn.Linear(4, 4)
    opt = torch.optim.SGD(model.parameters(), lr=1.0)
    sched = build_warmup_cosine_scheduler(opt, warmup_steps=5, total_steps=20, min_lr_ratio=0.1)
    seen = []
    for _ in range(20):
        seen.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert seen[0] < seen[4]  # warming up
    assert seen[-1] < seen[5]  # decayed below the post-warmup peak
    assert seen[-1] >= 0.1 - 1e-6  # floor respected


def test_ema_tracks_and_converges_to_param_when_decay_zero():
    model = nn.Linear(3, 3)
    ema = EMA(model, decay=0.0)  # decay=0 -> shadow instantly equals live params
    with torch.no_grad():
        model.weight.fill_(5.0)
    ema.update(model)
    assert torch.allclose(ema.shadow["weight"], model.weight)


def test_ema_high_decay_moves_slowly_towards_param():
    model = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    ema = EMA(model, decay=0.99)
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema.update(model)
    # after one update with high decay, the shadow should have moved only a
    # little bit towards the live (target) value, not jumped all the way
    assert 0.0 < ema.shadow["weight"].mean().item() < 0.05


def test_ema_copy_to_overwrites_model():
    model = nn.Linear(2, 2, bias=False)
    shadow_model = nn.Linear(2, 2, bias=False)
    ema = EMA(model, decay=0.999)
    with torch.no_grad():
        shadow_model.weight.fill_(3.14)
    ema.load_state_dict(shadow_model.state_dict())
    ema.copy_to(model)
    assert torch.allclose(model.weight, shadow_model.weight)


def test_encoder_dropout_and_drop_path_are_wired_and_deterministic_in_eval():
    torch.manual_seed(0)
    enc = EquivariantEncoder(hidden_dim=16, num_layers=3, dropout=0.5, drop_path_max=0.5)
    z = torch.randint(1, 10, (6,))
    pos = torch.randn(6, 3)
    batch = torch.tensor([0, 0, 0, 1, 1, 1])

    enc.eval()
    out1 = enc(z, pos, batch)
    out2 = enc(z, pos, batch)
    # eval mode: dropout/drop-path must be no-ops -> fully deterministic
    assert torch.allclose(out1.h, out2.h)
    assert torch.allclose(out1.v, out2.v)

    enc.train()
    torch.manual_seed(1)
    out_a = enc(z, pos, batch)
    torch.manual_seed(2)
    out_b = enc(z, pos, batch)
    # train mode with dropout=0.5: different RNG streams should (almost
    # certainly) give different activations
    assert not torch.allclose(out_a.h, out_b.h)


def test_encoder_default_zero_dropout_matches_no_regularisation_path():
    # Backwards-compatible default: dropout=0, drop_path_max=0 behaves exactly
    # like eval-mode regardless of train()/eval(), since both are disabled.
    torch.manual_seed(0)
    enc = EquivariantEncoder(hidden_dim=8, num_layers=2)
    z = torch.randint(1, 10, (4,))
    pos = torch.randn(4, 3)
    batch = torch.zeros(4, dtype=torch.long)
    enc.train()
    out1 = enc(z, pos, batch)
    out2 = enc(z, pos, batch)
    assert torch.allclose(out1.h, out2.h)
