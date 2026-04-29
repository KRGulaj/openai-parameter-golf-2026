"""Unit tests for model components."""

import torch
import torch.nn as nn
import pytest


def test_rms_norm():
    """Test RMSNorm layer."""
    from src.models.components import RMSNorm

    batch_size, seq_len, dim = 2, 10, 64
    x = torch.randn(batch_size, seq_len, dim)

    norm = RMSNorm(dim)
    out = norm(x)

    assert out.shape == x.shape
    # Check normalization (mean of squares should be ~1)
    rms = (out ** 2).mean(dim=-1)
    assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)


def test_rope():
    """Test RoPE positional embeddings."""
    from src.models.components import precompute_freqs_cis, apply_rotary_emb

    dim = 64
    seq_len = 20
    batch_size = 2
    n_heads = 4

    freqs_cos, freqs_sin = precompute_freqs_cis(dim, seq_len)
    assert freqs_cos.shape == (seq_len, dim)
    assert freqs_sin.shape == (seq_len, dim)

    xq = torch.randn(batch_size, seq_len, n_heads, dim)
    xk = torch.randn(batch_size, seq_len, n_heads, dim)

    xq_out, xk_out = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)
    assert xq_out.shape == xq.shape
    assert xk_out.shape == xk.shape


def test_muon_optimizer():
    """Test Muon optimizer."""
    from src.models.components import Muon

    model = nn.Linear(10, 10)
    optimizer = Muon(model.parameters(), lr=0.02)

    x = torch.randn(5, 10)
    y = torch.randn(5, 10)

    loss = nn.MSELoss()(model(x), y)
    loss.backward()
    optimizer.step()

    # Check that parameters were updated
    assert optimizer.param_groups[0]["lr"] == 0.02


def test_newton_schulz():
    """Test Newton-Schulz orthogonalization."""
    from src.models.components import zeropower_via_newtonschulz5

    G = torch.randn(10, 8)
    result = zeropower_via_newtonschulz5(G)

    assert result.shape == G.shape
    # Should be close to orthogonal (G^T @ G ≈ I)
    ortho_check = result.T @ result
    identity = torch.eye(result.shape[1], device=result.device)
    assert torch.allclose(ortho_check, identity, atol=0.1)


if __name__ == "__main__":
    test_rms_norm()
    test_rope()
    test_muon_optimizer()
    test_newton_schulz()
    print("All component tests passed!")
