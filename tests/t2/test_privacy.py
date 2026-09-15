import torch
from torch.utils.data import DataLoader, TensorDataset

from src.t2.config import delta_for_client
from src.t2.model import ResolutionMLP, regression_loss
from src.t2.privacy import make_training_state


def test_delta_rule_is_strictly_below_inverse_n():
    for n in (10, 1000, 1_000_000):
        delta = delta_for_client(n)
        assert 0 < delta < 1 / n


def test_private_accountant_persists_and_matches_executed_steps():
    torch.manual_seed(7)
    x = torch.randn(32, 2)
    y = torch.randn(32)
    loader = DataLoader(TensorDataset(x, y), batch_size=8, shuffle=True)
    state = make_training_state(
        model=ResolutionMLP(2, (8,)),
        loader=loader,
        mode="private",
        target_epsilon=5.0,
        learning_rate=0.01,
        max_grad_norm=1.0,
        planned_steps=8,
    )
    eps = []
    for round_number in (1, 2):
        for bx, by in state.loader:
            state.optimizer.zero_grad(set_to_none=True)
            loss = regression_loss(state.model(bx), by)
            loss.backward()
            state.optimizer.step()
            state.steps_taken += 1
        snap = state.snapshot(round_number)
        assert snap["accounted_steps"] == snap["steps_taken"]
        eps.append(float(snap["epsilon"]))
    assert state.steps_taken == 8
    assert state.accountant_steps() == 8
    assert eps[1] >= eps[0] > 0
    report = state.report(epsilon_tolerance=0.05)
    assert report["target_achieved_within_tolerance"] is True


def test_zero_noise_control_is_explicitly_nonprivate():
    x = torch.randn(16, 2)
    y = torch.randn(16)
    loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=True)
    state = make_training_state(
        model=ResolutionMLP(2, (4,)), loader=loader,
        mode="clipped_no_noise", target_epsilon=float("inf"),
        learning_rate=0.01, max_grad_norm=1.0, planned_steps=4,
    )
    report = state.report(epsilon_tolerance=0.05)
    assert report["mode"] == "clipped_no_noise"
    assert report["accountant"] is None
    assert report["realized_epsilon"] == float("inf")
    assert "non-private" in report["mechanism"]
