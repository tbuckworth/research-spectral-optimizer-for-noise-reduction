import copy
import math

import pytest
import torch

from numerai_competitive.filters import OPERATION_ORDER, StreamingSpectralFilter


def _model():
    return torch.nn.Sequential(torch.nn.Linear(5, 4), torch.nn.Tanh(), torch.nn.Linear(4, 2))


def _gradient(filter_, gradient):
    output, diagnostics = filter_.filter_flat(gradient.clone())
    return output, diagnostics


def test_strength_zero_is_exact_gradient_and_update_equivalent():
    torch.manual_seed(2)
    control, treated = _model(), _model()
    treated.load_state_dict(control.state_dict())
    opt_control = torch.optim.AdamW(control.parameters(), lr=1e-3, weight_decay=0.03)
    opt_treated = torch.optim.AdamW(treated.parameters(), lr=1e-3, weight_decay=0.03)
    spectral = StreamingSpectralFilter(treated, rank=5, warmup=0, strength=0.0)

    for _ in range(8):
        x, y = torch.randn(11, 5), torch.randn(11, 2)
        opt_control.zero_grad(); opt_treated.zero_grad()
        torch.nn.functional.mse_loss(control(x), y).backward()
        torch.nn.functional.mse_loss(treated(x), y).backward()
        before = torch.cat([p.grad.flatten().clone() for p in treated.parameters()])
        diagnostics = spectral.filter_grad()
        after = torch.cat([p.grad.flatten() for p in treated.parameters()])
        torch.testing.assert_close(after, before, rtol=0, atol=0)
        opt_control.step(); opt_treated.step()
    for left, right in zip(control.parameters(), treated.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=1e-6)
    assert not diagnostics["filtering_active"]


@pytest.mark.parametrize("mode", ["learned", "random", "shuffled", "norm_matched"])
def test_basis_is_orthonormal_and_diagnostics_are_finite(mode):
    model = _model()
    filter_ = StreamingSpectralFilter(
        model, rank=7, warmup=1, mode=mode, seed=91, history_size=5
    )
    generator = torch.Generator().manual_seed(7)
    for _ in range(30):
        gradient = torch.randn(filter_.n_params, generator=generator)
        output, diagnostics = _gradient(filter_, gradient)
        assert torch.isfinite(output).all()
        assert all(
            math.isfinite(float(value))
            for value in diagnostics.values()
            if isinstance(value, (int, float))
        )
    basis = filter_.random_basis[:, : filter_.V.shape[1]] if mode == "random" else filter_.V
    error = torch.linalg.matrix_norm(
        basis.T @ basis - torch.eye(basis.shape[1]), ord=2
    ).item()
    assert error <= 1e-5
    assert diagnostics["orthogonality_error"] <= 1e-5


@pytest.mark.parametrize("mode", ["learned", "random", "shuffled", "norm_matched"])
def test_state_restart_is_deterministic(mode):
    first = StreamingSpectralFilter(
        _model(), rank=6, warmup=0, mode=mode, seed=123, history_size=4, update_every=2
    )
    gradients = [torch.randn(first.n_params, generator=torch.Generator().manual_seed(i)) for i in range(12)]
    for gradient in gradients[:7]:
        _gradient(first, gradient)
    checkpoint = copy.deepcopy(first.state_dict())

    restarted = StreamingSpectralFilter(
        _model(), rank=6, warmup=0, mode=mode, seed=123, history_size=4, update_every=2
    )
    restarted.load_state_dict(checkpoint)
    for gradient in gradients[7:]:
        expected, expected_diagnostics = _gradient(first, gradient)
        actual, actual_diagnostics = _gradient(restarted, gradient)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert actual_diagnostics == expected_diagnostics
    torch.testing.assert_close(restarted.V, first.V, rtol=0, atol=0)
    torch.testing.assert_close(restarted.eigenvalues, first.eigenvalues, rtol=0, atol=0)


def test_controls_have_declared_semantics_and_ordering():
    assert "clipping" in OPERATION_ORDER
    assert OPERATION_ORDER.index("covariance") < OPERATION_ORDER.index("Adam moments")
    model = _model()
    learned = StreamingSpectralFilter(model, rank=4, warmup=0, mode="learned")
    norm_matched = StreamingSpectralFilter(_model(), rank=4, warmup=0, mode="norm_matched")
    gradients = [torch.arange(1, learned.n_params + 1, dtype=torch.float32)]
    gradients += [torch.roll(gradients[0], i) for i in range(1, 6)]
    for gradient in gradients:
        learned_output, _ = _gradient(learned, gradient)
        matched_output, _ = _gradient(norm_matched, gradient)
    torch.testing.assert_close(matched_output.norm(), learned_output.norm(), rtol=1e-6, atol=1e-6)
    cosine = torch.nn.functional.cosine_similarity(matched_output, gradients[-1], dim=0)
    torch.testing.assert_close(cosine, torch.tensor(1.0), rtol=1e-6, atol=1e-6)


def test_in_place_rank_one_accumulation_matches_outer_sum():
    generator = torch.Generator().manual_seed(44)
    basis_product = torch.randn(19, 7, generator=generator)
    direction = torch.randn(19, generator=generator)
    rotation = torch.randn(7, generator=generator)
    expected = basis_product + direction.unsqueeze(1) * rotation.unsqueeze(0)
    actual = basis_product.clone()
    actual.addr_(direction, rotation)
    torch.testing.assert_close(actual, expected)
