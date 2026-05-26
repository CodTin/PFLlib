from pathlib import Path

import typer
from typing_extensions import Annotated

from pfllib.config import ExperimentConfig

app = typer.Typer(name="pfllib", help="Personalized Federated Learning Library", no_args_is_help=True)


@app.command()
def run(
    dataset: Annotated[str, typer.Option("-d", "--dataset", help="Dataset name")] = "MNIST",
    algorithm: Annotated[str, typer.Option("-a", "--algorithm", help="FL algorithm")] = "FedAvg",
    model: Annotated[str, typer.Option("-m", "--model", help="Model architecture")] = "CNN",
    num_classes: Annotated[int, typer.Option(help="Number of classes")] = 10,
    num_clients: Annotated[int, typer.Option("-nc", help="Total number of clients")] = 20,
    global_rounds: Annotated[int, typer.Option("-gr", help="Global communication rounds")] = 2000,
    local_epochs: Annotated[int, typer.Option("-le", help="Local epochs per round")] = 1,
    batch_size: Annotated[int, typer.Option("-lbs", help="Batch size")] = 10,
    lr: Annotated[float, typer.Option("-lr", help="Local learning rate")] = 0.005,
    device: Annotated[str, typer.Option("-dev", help="Device (cpu/cuda)")] = "cuda",
    device_id: Annotated[str, typer.Option("-did", help="CUDA device ID")] = "0",
    join_ratio: Annotated[float, typer.Option("-jr", help="Client join ratio")] = 1.0,
    eval_gap: Annotated[int, typer.Option("-eg", help="Eval gap in rounds")] = 1,
    times: Annotated[int, typer.Option("-t", help="Number of runs")] = 1,
    goal: Annotated[str, typer.Option("-go", help="Experiment goal tag")] = "test",
    mu: Annotated[float, typer.Option(help="Proximal term weight")] = 0.0,
    beta: Annotated[float, typer.Option(help="Beta parameter")] = 0.0,
    lamda: Annotated[float, typer.Option(help="Lambda regularization")] = 1.0,
    alpha: Annotated[float, typer.Option(help="Alpha parameter")] = 1.0,
    tau: Annotated[float, typer.Option(help="Temperature parameter")] = 1.0,
    data_dir: Annotated[str, typer.Option(help="Data directory")] = "data",
    results_dir: Annotated[str, typer.Option(help="Results directory")] = "results",
    outputs_dir: Annotated[str, typer.Option(help="Outputs directory")] = "outputs",
    log_level: Annotated[str, typer.Option(help="Logging level")] = "INFO",
):
    """Run a federated learning experiment."""
    from pfllib.runner import run_experiment

    config = ExperimentConfig(
        dataset=dataset,
        algorithm=algorithm,
        model=model,
        num_classes=num_classes,
        num_clients=num_clients,
        global_rounds=global_rounds,
        local_epochs=local_epochs,
        batch_size=batch_size,
        local_learning_rate=lr,
        device=device,
        device_id=device_id,
        join_ratio=join_ratio,
        eval_gap=eval_gap,
        times=times,
        goal=goal,
        mu=mu,
        beta=beta,
        lamda=lamda,
        alpha=alpha,
        tau=tau,
        data_dir=Path(data_dir),
        results_dir=Path(results_dir),
        outputs_dir=Path(outputs_dir),
    )
    run_experiment(config)


@app.command(name="generate-data")
def generate_data(
    dataset: Annotated[str, typer.Argument(help="Dataset name (e.g. MNIST, Cifar10)")],
    num_clients: Annotated[int, typer.Option("-nc", help="Number of clients")] = 20,
    noniid: Annotated[bool, typer.Option(help="Non-IID partition")] = False,
    balance: Annotated[bool, typer.Option(help="Balanced partition")] = True,
    partition: Annotated[str, typer.Option(help="Partition strategy: dir/pat/exdir")] = "dir",
    data_dir: Annotated[str, typer.Option(help="Data directory")] = "data",
):
    """Generate a federated dataset."""
    import importlib

    dataset_lower = dataset.lower()
    try:
        mod = importlib.import_module(f"pfllib.data.generators.{dataset_lower}")
    except ImportError:
        typer.echo(f"Unknown dataset: {dataset}")
        raise typer.Exit(1)

    gen_fn = getattr(mod, "generate_dataset", None)
    if gen_fn is None:
        typer.echo(f"No generate_dataset function in {dataset_lower}")
        raise typer.Exit(1)

    import inspect

    sig = inspect.signature(gen_fn)
    params = list(sig.parameters.keys())

    kwargs: dict = {}
    if "dir_path" in params:
        kwargs["dir_path"] = f"{data_dir}/{dataset}/"
    if "num_clients" in params:
        kwargs["num_clients"] = num_clients
    if "niid" in params:
        kwargs["niid"] = noniid
    if "balance" in params:
        kwargs["balance"] = balance
    if "partition" in params:
        kwargs["partition"] = partition

    gen_fn(**kwargs)
    typer.echo(f"Dataset {dataset} generated successfully.")


@app.command(name="list")
def list_available(
    algorithms: Annotated[bool, typer.Option("--algorithms", "-a", help="List algorithms")] = False,
    models: Annotated[bool, typer.Option("--models", "-m", help="List models")] = False,
):
    """List available algorithms and models."""
    _ensure_registered()
    from pfllib.registry import list_algorithms, list_models

    if algorithms:
        for name in list_algorithms():
            typer.echo(name)
    if models:
        for name in list_models():
            typer.echo(name)
    if not algorithms and not models:
        typer.echo("Algorithms:")
        for name in list_algorithms():
            typer.echo(f"  {name}")
        typer.echo("\nModels:")
        for name in list_models():
            typer.echo(f"  {name}")


def _ensure_registered():
    import pfllib.clients  # noqa: F401
    import pfllib.models  # noqa: F401
    import pfllib.servers  # noqa: F401
