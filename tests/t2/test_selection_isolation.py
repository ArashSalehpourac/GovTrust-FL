import inspect

from src.t2.federated import train_select
from src.t2.runner import run_one


def test_train_select_has_no_external_data_parameter():
    params = set(inspect.signature(train_select).parameters)
    assert params == {"source_splits", "preprocessor", "config"}


def test_runner_loads_external_only_after_source_selection():
    source = inspect.getsource(run_one)
    selection = source.index("selected = train_select")
    external_load = source.index("external_raw, external_sha256 = external_loader()")
    assert selection < external_load
