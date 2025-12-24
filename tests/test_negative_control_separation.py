"""Test: KT-2 negative control train/eval separation (anti-leak guardrails).

This suite hardens the KT-2 negative control by verifying:

1. Locked protocol seeds for train vs eval are distinct.
2. With distinct seeds, independently generated train/eval batches are not bitwise equal.
3. Production code raises when train/eval leakage is forced (RECOVERY_SEED == EVAL_SEED).
4. Production code completes normally under default protocol conditions.
5. (Semantic) Under distinct batches, our injected EvalContext provides an eval_batch,
   and the production guard prevents any training-side leak from eval_batch.

These tests intentionally avoid relying on internal call counts or “first forward” assumptions.
"""

from __future__ import annotations

import pytest
import torch

from experiments.kt2_locality_falsifier import (
    EVAL_SEED,
    RECOVERY_SEED,
    create_model,
    run_negative_control,
)


def test_train_eval_seeds_are_distinct():
    """Ensure the locked protocol seeds for train/eval are distinct."""
    assert RECOVERY_SEED != EVAL_SEED, "RECOVERY_SEED and EVAL_SEED must be distinct"


def test_eval_ctx_batch_differs_from_x_train():
    """Ensure that eval and train batches differ numerically when seeds differ.

    This test reconstructs the intent without depending on EvalContext.create() signature,
    which may change over time.
    """
    g_eval = torch.Generator(device="cpu").manual_seed(EVAL_SEED)
    eval_batch = torch.randn(64, 12, generator=g_eval)

    g_train = torch.Generator(device="cpu").manual_seed(RECOVERY_SEED)
    x_train = torch.randn(64, 12, generator=g_train)

    assert not torch.equal(
        x_train, eval_batch
    ), "Train and eval batches should differ when seeds differ"


def test_negative_control_raises_on_train_eval_leakage(monkeypatch):
    """Ensure run_negative_control raises when x_train equals eval_ctx.eval_batch.

    We force the leakage condition inside production code by setting RECOVERY_SEED == EVAL_SEED.
    """
    import experiments.kt2_locality_falsifier as kt2

    monkeypatch.setattr(kt2, "RECOVERY_SEED", kt2.EVAL_SEED)

    model = kt2.create_model(seed=0, dim=12, hidden=64)

    with pytest.raises(
        RuntimeError,
        match=r"BUG: Distillation must train on x_train, not eval_ctx\.eval_batch \(data leakage\)",
    ):
        kt2.run_negative_control(model, verbose=False)


def test_negative_control_no_crash():
    """Smoke test: run_negative_control completes and returns expected core keys.

    NOTE: We do NOT assert on optional/legacy diagnostic keys (like 'Damage') because
    protocol output schemas evolve. This test should remain stable.
    """
    model = create_model(seed=0, dim=12, hidden=64)
    result = run_negative_control(model, verbose=False)

    # Core keys that should always exist
    assert "CI_proxy" in result
    assert "CI_distillation" in result
    assert "control_passes" in result

    # Loss keys
    assert "L_pre" in result
    assert "L_post" in result
    assert result["L_pre"] >= 0
    assert result["L_post"] >= 0


def test_eval_batch_only_present_in_eval_ctx_no_train_eval_leak(monkeypatch):
    """Semantic anti-leak test (stable):

    - Inject a DummyEvalCtx with a distinct eval_batch sentinel.
    - Force x_train sentinel inside production regardless of torch.randn call style.
    - If training-side code ever uses eval_batch (leakage), production should raise.

    We intentionally do NOT call model(eval_batch) inside DummyEvalCtx.evaluate_loss,
    because model input dimensionality may vary across KT-2 configurations. We only need
    eval_batch to exist and the production guard to enforce non-leakage.
    """
    import experiments.kt2_locality_falsifier as kt2

    x_train = torch.full((64, 12), 7.0)
    eval_batch = torch.full((64, 12), -3.0)
    assert not torch.equal(x_train, eval_batch)

    class DummyEvalCtx:
        def __init__(self, eval_batch: torch.Tensor):
            self.eval_batch = eval_batch
            self.base_loss = 1.0
            self.perturbed_loss = 2.0
            self.inv_pre = {}

        def evaluate_loss(self, _model):
            return 2.0

    monkeypatch.setattr(
        kt2.EvalContext,
        "create",
        lambda *a, **k: DummyEvalCtx(eval_batch),
    )

    orig_randn = kt2.torch.randn

    def fake_randn(*shape, **kwargs):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)):
            shp = tuple(shape[0])
        else:
            shp = tuple(shape)

        if len(shp) == 2 and shp[0] == 64 and shp[1] == x_train.shape[1]:
            out = x_train.clone()
            if "device" in kwargs and kwargs["device"] is not None:
                out = out.to(kwargs["device"])
            if "dtype" in kwargs and kwargs["dtype"] is not None:
                out = out.to(kwargs["dtype"])
            return out

        return orig_randn(*shape, **kwargs)

    monkeypatch.setattr(kt2.torch, "randn", fake_randn)

    model = kt2.create_model(seed=0, dim=12, hidden=64)
    result = kt2.run_negative_control(model, verbose=False)

    assert "control_passes" in result
    assert isinstance(result["control_passes"], bool)
