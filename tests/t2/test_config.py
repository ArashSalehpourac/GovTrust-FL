import pytest

from src.t2.config import T2Config, diagnostic_plan


def test_diagnostic_plan_is_only_boston_la_eps_5_1_inf_seeds_0_1_2():
    plan = diagnostic_plan()
    assert len(plan) == 24  # 2 held-outs x 3 seeds x (inf,5,1,matched-control)
    assert {row["held_out_city"] for row in plan} == {"boston", "los_angeles"}
    assert {row["seed"] for row in plan} == {0, 1, 2}
    private = [row for row in plan if row["mode"] == "private"]
    assert {row["epsilon"] for row in private} == {1.0, 5.0}


def test_fedprox_is_blocked_from_inferential_t2_config():
    with pytest.raises(ValueError):
        T2Config(algorithm="fedprox").validate()  # type: ignore[arg-type]
