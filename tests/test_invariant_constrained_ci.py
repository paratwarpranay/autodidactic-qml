"""Tests for invariant-constrained CI metric.

Key properties tested:
1. Deterministic perturbation (same seed → same result)
2. Deterministic evaluation (same eval_seed → same PRE/POST loss)
3. Basin proxy invariant: constraint never makes inv_recovery_ratio worse
4. Differentiability check: penalty gradients flow
5. Sanity assertions enforce determinism
6. Spectral entropy shape penalty
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from ucip_detection.invariant_constrained_ci import (
    InvariantConstrainedCI,
    apply_perturbation,
    PerturbationConfig,
    EvalContext,
    EvalSuite,
    compute_model_invariants,
    invariant_distance,
    compare_recovery_methods,
    sweep_invariant_weight,
    _find_weight_param,
    _spectral_entropy_torch,
    PENALTY_KEYS_SCALE,
    PENALTY_KEYS_SHAPE,
)


class SimpleRNN(nn.Module):
    """Minimal RNN for testing."""
    def __init__(self, hidden_size=16):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = hidden_size
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.1)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x, steps=1):
        h = self.fc1(x)
        for _ in range(steps):
            h = torch.tanh(h @ self.W_hh)
        return self.fc2(h), {"h_final": h}


class TestPerturbationDeterminism:
    """Test that perturbation is reproducible with same seed."""
    
    def test_same_seed_same_result(self):
        model = SimpleRNN(hidden_size=16)
        config = PerturbationConfig(seed=12345, strength=0.25, zero_frac=0.05)
        
        pert1 = apply_perturbation(model, config)
        pert2 = apply_perturbation(model, config)
        
        w1 = list(pert1.parameters())[0].detach().numpy()
        w2 = list(pert2.parameters())[0].detach().numpy()
        
        np.testing.assert_array_almost_equal(w1, w2)
    
    def test_different_seed_different_result(self):
        model = SimpleRNN(hidden_size=16)
        config1 = PerturbationConfig(seed=12345, strength=0.25, zero_frac=0.05)
        config2 = PerturbationConfig(seed=54321, strength=0.25, zero_frac=0.05)
        
        pert1 = apply_perturbation(model, config1)
        pert2 = apply_perturbation(model, config2)
        
        w1 = list(pert1.parameters())[0].detach().numpy()
        w2 = list(pert2.parameters())[0].detach().numpy()
        
        assert not np.allclose(w1, w2)


class TestEvalContextDeterminism:
    """Test that evaluation context produces deterministic measurements."""
    
    def test_same_eval_seed_same_losses(self):
        model = SimpleRNN(hidden_size=16)
        config = PerturbationConfig(seed=42, strength=0.25, zero_frac=0.05)
        perturbed = apply_perturbation(model, config)
        
        ctx1 = EvalContext.create(model, perturbed, eval_seed=12345)
        ctx2 = EvalContext.create(model, perturbed, eval_seed=12345)
        
        assert ctx1.base_loss == ctx2.base_loss
        assert ctx1.perturbed_loss == ctx2.perturbed_loss
        assert ctx1.d_post == ctx2.d_post
    
    def test_different_eval_seed_different_losses(self):
        model = SimpleRNN(hidden_size=16)
        config = PerturbationConfig(seed=42, strength=0.25, zero_frac=0.05)
        perturbed = apply_perturbation(model, config)
        
        ctx1 = EvalContext.create(model, perturbed, eval_seed=12345)
        ctx2 = EvalContext.create(model, perturbed, eval_seed=54321)
        
        assert ctx1.base_loss != ctx2.base_loss
    
    def test_evaluate_loss_deterministic(self):
        model = SimpleRNN(hidden_size=16)
        config = PerturbationConfig(seed=42, strength=0.25, zero_frac=0.05)
        perturbed = apply_perturbation(model, config)
        
        ctx = EvalContext.create(model, perturbed, eval_seed=12345)
        
        loss1 = ctx.evaluate_loss(model)
        loss2 = ctx.evaluate_loss(model)
        
        assert loss1 == loss2


class TestEvalSuite:
    """Test EvalSuite for mean±std reporting."""
    
    def test_suite_creates_multiple_contexts(self):
        model = SimpleRNN(hidden_size=16)
        config = PerturbationConfig(seed=42, strength=0.25, zero_frac=0.05)
        perturbed = apply_perturbation(model, config)
        
        suite = EvalSuite.create(model, perturbed, n_batches=5)
        
        assert len(suite.contexts) == 5
    
    def test_suite_contexts_have_different_batches(self):
        model = SimpleRNN(hidden_size=16)
        config = PerturbationConfig(seed=42, strength=0.25, zero_frac=0.05)
        perturbed = apply_perturbation(model, config)
        
        suite = EvalSuite.create(model, perturbed, n_batches=3, base_eval_seed=100)
        
        losses = [c.base_loss for c in suite.contexts]
        assert len(set(losses)) == 3
    
    def test_suite_mean_std(self):
        model = SimpleRNN(hidden_size=16)
        config = PerturbationConfig(seed=42, strength=0.25, zero_frac=0.05)
        perturbed = apply_perturbation(model, config)
        
        suite = EvalSuite.create(model, perturbed, n_batches=5)
        
        assert suite.base_loss_mean > 0
        assert suite.base_loss_std >= 0
        assert suite.perturbed_loss_mean > 0
        assert suite.d_post_mean > 0


class TestRecoverySeed:
    """Test recovery_seed for deterministic vs stochastic recovery."""
    
    def test_same_recovery_seed_same_result(self):
        model = SimpleRNN(hidden_size=16)
        
        config = PerturbationConfig(seed=42, strength=0.25, zero_frac=0.05)
        perturbed = apply_perturbation(model, config)
        eval_ctx = EvalContext.create(model, perturbed, eval_seed=12345)
        
        metric1 = InvariantConstrainedCI(recovery_seed=999, recovery_steps=5)
        metric2 = InvariantConstrainedCI(recovery_seed=999, recovery_steps=5)
        
        result1 = metric1.score(model, eval_context=eval_ctx, perturbed_model=perturbed)
        result2 = metric2.score(model, eval_context=eval_ctx, perturbed_model=perturbed)
        
        assert result1["ci_score"] == result2["ci_score"]
        assert result1["recovered_loss"] == result2["recovered_loss"]
    
    def test_different_recovery_seed_different_result(self):
        model = SimpleRNN(hidden_size=16)
        
        config = PerturbationConfig(seed=42, strength=0.25, zero_frac=0.05)
        perturbed = apply_perturbation(model, config)
        eval_ctx = EvalContext.create(model, perturbed, eval_seed=12345)
        
        metric1 = InvariantConstrainedCI(recovery_seed=999, recovery_steps=5)
        metric2 = InvariantConstrainedCI(recovery_seed=888, recovery_steps=5)
        
        result1 = metric1.score(model, eval_context=eval_ctx, perturbed_model=perturbed)
        result2 = metric2.score(model, eval_context=eval_ctx, perturbed_model=perturbed)
        
        assert result1["recovered_loss"] != result2["recovered_loss"]


class TestBasinProxy:
    """Test the basin proxy property."""
    
    def test_constraint_improves_or_maintains_inv_ratio(self):
        model = SimpleRNN(hidden_size=16)
        
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        for _ in range(20):
            x = torch.randn(32, 16)
            y, _ = model(x)
            loss = torch.mean((y - x) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        config = PerturbationConfig(seed=42, strength=0.25, zero_frac=0.05)
        perturbed = apply_perturbation(model, config)
        eval_ctx = EvalContext.create(model, perturbed, eval_seed=12345)
        
        metric_0 = InvariantConstrainedCI(recovery_steps=5, recovery_seed=42)
        result_0 = metric_0.score(model, use_invariant_constraint=False, 
                                  eval_context=eval_ctx, perturbed_model=perturbed)
        
        metric_λ = InvariantConstrainedCI(invariant_weight=0.1, recovery_steps=5, recovery_seed=42)
        result_λ = metric_λ.score(model, use_invariant_constraint=True,
                                  eval_context=eval_ctx, perturbed_model=perturbed)
        
        tolerance = 0.05
        assert result_λ["inv_recovery_ratio"] <= result_0["inv_recovery_ratio"] + tolerance


class TestDifferentiability:
    """Test that invariant penalty is truly differentiable."""
    
    def test_penalty_has_gradient(self):
        model = SimpleRNN(hidden_size=16)
        target_inv = compute_model_invariants(model)
        
        config = PerturbationConfig(seed=42, strength=0.5, zero_frac=0.0)
        perturbed = apply_perturbation(model, config)
        
        metric = InvariantConstrainedCI()
        penalty = metric._invariant_penalty_torch(perturbed, target_inv)
        
        assert penalty.item() > 0.01
        
        penalty.backward()
        
        W = _find_weight_param(perturbed)
        assert W is not None
        assert W.grad is not None
        assert torch.abs(W.grad).sum() > 0


class TestSpectralEntropy:
    """Test differentiable spectral entropy."""
    
    def test_spectral_entropy_differentiable(self):
        """Spectral entropy should be differentiable."""
        M = torch.randn(8, 8, requires_grad=True)
        M_sym = (M + M.T) / 2
        
        entropy = _spectral_entropy_torch(M_sym)
        entropy.backward()
        
        assert M.grad is not None
        assert torch.abs(M.grad).sum() > 0
    
    def test_spectral_entropy_uniform_is_max(self):
        """Uniform eigenvalues should give maximum entropy."""
        M = torch.eye(8)
        entropy_uniform = _spectral_entropy_torch(M)
        
        M_peaked = torch.diag(torch.tensor([10.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))
        entropy_peaked = _spectral_entropy_torch(M_peaked)
        
        assert entropy_uniform > entropy_peaked


class TestShapePenalty:
    """Test shape-sensitive penalty with spec_entropy."""
    
    def test_shape_penalty_includes_spec_entropy(self):
        """PENALTY_KEYS_SHAPE should include spec_entropy."""
        assert "spec_entropy" in PENALTY_KEYS_SHAPE
        assert "spec_entropy" not in PENALTY_KEYS_SCALE
    
    def test_shape_penalty_differentiable(self):
        """Shape penalty should be differentiable through spec_entropy."""
        model = SimpleRNN(hidden_size=16)
        target_inv = compute_model_invariants(model)
        
        config = PerturbationConfig(seed=42, strength=0.5, zero_frac=0.0)
        perturbed = apply_perturbation(model, config)
        
        metric = InvariantConstrainedCI(penalty_keys=PENALTY_KEYS_SHAPE)
        penalty = metric._invariant_penalty_torch(perturbed, target_inv)
        
        assert penalty.item() > 0.01
        penalty.backward()
        
        W = _find_weight_param(perturbed)
        assert W.grad is not None


class TestGradientRatio:
    """Test gradient ratio diagnostic."""
    
    def test_gradient_ratio_computed(self):
        """Gradient ratio should be computed and returned."""
        model = SimpleRNN(hidden_size=16)
        
        metric = InvariantConstrainedCI(
            recovery_steps=3, 
            track_gradient_ratio=True,
            invariant_weight=0.1,
        )
        result = metric.score(model, use_invariant_constraint=True)
        
        assert "gradient_ratio" in result
        assert result["gradient_ratio"] is not None
        assert result["gradient_ratio"] > 0
    
    def test_gradient_ratio_none_for_unconstrained(self):
        """Gradient ratio should be None for unconstrained recovery."""
        model = SimpleRNN(hidden_size=16)
        
        metric = InvariantConstrainedCI(recovery_steps=3)
        result = metric.score(model, use_invariant_constraint=False)
        
        assert result["gradient_ratio"] is None


class TestCompareRecoveryMethods:
    """Test comparison function with sanity assertions."""
    
    def test_pre_post_losses_match_exactly(self):
        """PRE and POST losses must be bit-for-bit identical."""
        model = SimpleRNN(hidden_size=16)
        
        comparison = compare_recovery_methods(
            model, 
            verbose=False, 
            recovery_steps=3,
            perturb_seed=42,
            eval_seed=12345,
            recovery_seed=999,
        )
        
        pre_std = comparison["standard"]["base_loss"]
        pre_con = comparison["constrained"]["base_loss"]
        assert pre_std == pre_con
        
        post_std = comparison["standard"]["perturbed_loss"]
        post_con = comparison["constrained"]["perturbed_loss"]
        assert post_std == post_con
        
        d_post_std = comparison["standard"]["d_post"]
        d_post_con = comparison["constrained"]["d_post"]
        assert d_post_std == d_post_con
        
        assert comparison["determinism_verified"] is True
    
    def test_returns_comparison_metrics(self):
        model = SimpleRNN(hidden_size=16)
        
        comparison = compare_recovery_methods(model, verbose=False, recovery_steps=2)
        
        assert "ci_improvement" in comparison
        assert "inv_improvement" in comparison
        assert "determinism_verified" in comparison


class TestSweepDeterminism:
    """Test that sweep maintains determinism across all λ."""
    
    def test_all_lambda_have_same_pre_post_d_post(self):
        model = SimpleRNN(hidden_size=16)
        
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        for _ in range(10):
            x = torch.randn(16, 16)
            y, _ = model(x)
            loss = torch.mean((y - x) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        results = sweep_invariant_weight(
            model,
            weights=[0.0, 0.1, 1.0],
            recovery_steps=3,
            perturb_seed=42,
            eval_seed=12345,
            recovery_seed=999,
        )
        
        d_posts = [r["d_post"] for r in results]
        assert all(dp == d_posts[0] for dp in d_posts)


class TestInvariantDistance:
    """Test invariant distance calculation."""
    
    def test_zero_distance_identical(self):
        inv = {"fro_norm": 2.5, "tr_M2": 4.0}
        d = invariant_distance(inv, inv)
        assert np.isclose(d, 0.0)
    
    def test_normalized_by_scale(self):
        inv1 = {"fro_norm": 100.0, "tr_M2": 100.0}
        inv2 = {"fro_norm": 101.0, "tr_M2": 101.0}
        
        d = invariant_distance(inv1, inv2)
        assert d < 0.1


class TestCIMetric:
    """Test CI metric computation."""
    
    def test_returns_required_fields(self):
        model = SimpleRNN(hidden_size=16)
        metric = InvariantConstrainedCI(recovery_steps=2)
        
        result = metric.score(model, use_invariant_constraint=True)
        
        required = [
            "base_loss", "perturbed_loss", "recovered_loss", "ci_score",
            "d_post", "d_rec", "inv_recovery_ratio",
            "perturbation_config", "eval_seed", "recovery_seed",
            "gradient_ratio", "penalty_keys",
        ]
        for key in required:
            assert key in result, f"Missing key: {key}"
    
    def test_ci_bounded(self):
        model = SimpleRNN(hidden_size=16)
        metric = InvariantConstrainedCI(recovery_steps=5)
        
        result = metric.score(model)
        
        assert -0.1 <= result["ci_score"] <= 2.0
