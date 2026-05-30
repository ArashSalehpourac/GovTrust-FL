"""Flower server helpers."""

try:
    import flwr as fl
except ImportError:  # pragma: no cover
    fl = None


def require_flower() -> None:
    """Raise a helpful error when Flower is not installed."""

    if fl is None:
        raise ImportError("Flower is not installed. Run `pip install flwr` first.")


def make_fedavg_strategy(**kwargs):
    """Create a standard FedAvg strategy."""

    require_flower()
    return fl.server.strategy.FedAvg(**kwargs)


def start_server(
    server_address: str = "0.0.0.0:8080",
    num_rounds: int = 5,
    strategy=None,
) -> None:
    """Start a Flower server for local federated experiments."""

    require_flower()
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy or make_fedavg_strategy(),
    )

