"""Tests for FinslerAdam optimizer."""

import pytest
import torch
import torch.nn as nn


class TestFinslerAdam:
    """Core optimizer tests."""

    def test_import(self):
        from finsler_adam import FinslerAdam
        assert FinslerAdam is not None

    def test_basic_step(self):
        from finsler_adam import FinslerAdam
        model = nn.Linear(10, 1)
        opt = FinslerAdam(model.parameters(), lr=1e-3)
        x = torch.randn(4, 10)
        loss = model(x).pow(2).mean()
        loss.backward()
        opt.step()

    def test_reduces_to_adamw(self):
        """gamma=0, anna_alpha=0 should behave identically to AdamW."""
        from finsler_adam import FinslerAdam
        torch.manual_seed(42)
        model_f = nn.Linear(5, 1)
        torch.manual_seed(42)
        model_a = nn.Linear(5, 1)

        opt_f = FinslerAdam(model_f.parameters(), lr=1e-2, gamma=0.0, anna_alpha=0.0,
                            weight_decay=0.01)
        opt_a = torch.optim.AdamW(model_a.parameters(), lr=1e-2, weight_decay=0.01)

        for _ in range(10):
            torch.manual_seed(_)
            x = torch.randn(8, 5)
            loss_f = model_f(x).pow(2).mean()
            loss_a = model_a(x).pow(2).mean()
            opt_f.zero_grad(); loss_f.backward(); opt_f.step()
            opt_a.zero_grad(); loss_a.backward(); opt_a.step()

        for pf, pa in zip(model_f.parameters(), model_a.parameters()):
            assert torch.allclose(pf, pa, atol=1e-5), "gamma=0, alpha=0 should match AdamW"

    def test_finsler_scaling_changes_behavior(self):
        """gamma>0 should produce different parameters than gamma=0."""
        from finsler_adam import FinslerAdam
        torch.manual_seed(42)
        model0 = nn.Linear(5, 1)
        torch.manual_seed(42)
        model1 = nn.Linear(5, 1)

        opt0 = FinslerAdam(model0.parameters(), lr=1e-2, gamma=0.0, anna_alpha=0.0)
        opt1 = FinslerAdam(model1.parameters(), lr=1e-2, gamma=0.5, anna_alpha=0.0)

        for _ in range(20):
            torch.manual_seed(_)
            x = torch.randn(8, 5)
            for m, o in [(model0, opt0), (model1, opt1)]:
                loss = m(x).pow(2).mean()
                o.zero_grad(); loss.backward(); o.step()

        params_differ = any(
            not torch.allclose(p0, p1, atol=1e-6)
            for p0, p1 in zip(model0.parameters(), model1.parameters())
        )
        assert params_differ, "gamma=0.5 should diverge from gamma=0"

    def test_anna_limit_preserves_small_grads(self):
        """Anna-Limit should have minimal impact on small gradients."""
        from finsler_adam.anna_limit import anna_clip
        g = torch.tensor([0.1, 0.2, 0.5])
        clipped = anna_clip(g, alpha=0.1)
        ratio = clipped / g
        assert (ratio > 0.9).all(), f"Small grads should be >90% preserved, got {ratio}"

    def test_anna_limit_clips_large_grads(self):
        """Anna-Limit should strongly clip large gradients."""
        from finsler_adam.anna_limit import anna_clip
        g = torch.tensor([100.0, 1000.0])
        clipped = anna_clip(g, alpha=0.1)
        ratio = (clipped / g).abs()
        assert (ratio < 0.1).all(), f"Large grads should be >90% clipped, got {ratio}"

    def test_no_nan_or_inf(self):
        """No NaN or Inf after many steps."""
        from finsler_adam import FinslerAdam
        model = nn.Linear(10, 1)
        opt = FinslerAdam(model.parameters(), lr=0.05, gamma=0.5, anna_alpha=0.1)
        for _ in range(200):
            x = torch.randn(8, 10)
            loss = model(x).pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        for p in model.parameters():
            assert torch.isfinite(p).all(), "Parameters should remain finite"

    def test_invalid_params_raise(self):
        from finsler_adam import FinslerAdam
        model = nn.Linear(2, 1)
        with pytest.raises(ValueError):
            FinslerAdam(model.parameters(), lr=-1)
        with pytest.raises(ValueError):
            FinslerAdam(model.parameters(), gamma=1.5)

    def test_sparse_grad_raises(self):
        from finsler_adam import FinslerAdam
        emb = nn.Embedding(10, 3, sparse=True)
        opt = FinslerAdam(emb.parameters(), lr=1e-3)
        loss = emb(torch.tensor([1, 2, 3])).sum()
        loss.backward()
        with pytest.raises(RuntimeError, match="sparse"):
            opt.step()
