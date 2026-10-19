"""Evaluate a saved CAPTCHA checkpoint on one or more datasets."""

import argparse
import json
import os
from pathlib import Path

import torch

from cs172.models import MODELS, create_model

if __package__:
    from .train import evaluate, make_loader
else:
    from train import evaluate, make_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="checkpoint produced by scripts/train.py",
    )
    parser.add_argument(
        "--data",
        nargs="+",
        required=True,
        help="one or more image directories or ZIP archives",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="run config (default: infer config.json from the checkpoint path)",
    )
    parser.add_argument("--model", choices=MODELS, help="override the model in config.json")
    parser.add_argument(
        "--image-size",
        nargs=2,
        type=int,
        metavar=("HEIGHT", "WIDTH"),
        help="override the image size in config.json",
    )
    parser.add_argument("--batch-size", type=int, help="evaluation batch size (default: value from config or 64)")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="evaluation device (default: CUDA when available, otherwise CPU)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("evaluation.json"),
        help="output path (default: evaluation.json in the experiment directory)",
    )
    args = parser.parse_args()
    validate_args(parser, args)
    return args


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    if args.config is not None and not args.config.is_file():
        parser.error(f"config does not exist: {args.config}")
    if args.image_size is not None and any(size < 1 for size in args.image_size):
        parser.error("--image-size values must be positive")
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.workers < 0:
        parser.error("--workers cannot be negative")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda was requested, but CUDA is not available")


def infer_config_path(checkpoint_path: Path) -> Path:
    """Infer ``<run-dir>/config.json`` from a checkpoint path."""
    candidates = (
        checkpoint_path.parent.parent / "config.json",
        checkpoint_path.parent / "config.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "could not infer config.json from the checkpoint path; pass it explicitly with --config"
    )


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        config = json.load(file)
    if not isinstance(config, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return config


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def resolve_output_path(path: Path, run_dir: Path) -> Path:
    """Resolve a report path, placing a bare filename inside the run directory."""
    path = path.expanduser()
    if not path.is_absolute() and path.parent == Path("."):
        path = run_dir / path
    return path.resolve()


def format_evaluation_metrics(name: str, metrics: dict[str, float]) -> str:
    return f"{name} digit {metrics['digit_acc']:.2%}, image {metrics['image_acc']:.2%}"


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    config_path = (args.config or infer_config_path(checkpoint_path)).expanduser().resolve()
    config = load_config(config_path)

    model_name = args.model or config.get("model")
    image_size_value = args.image_size or config.get("image_size")
    if model_name is None:
        raise ValueError("model is missing from config.json; pass --model")
    if image_size_value is None:
        raise ValueError("image_size is missing from config.json; pass --image-size HEIGHT WIDTH")

    image_size = tuple(image_size_value)
    if len(image_size) != 2 or any(size < 1 for size in image_size):
        raise ValueError(f"invalid image_size: {image_size_value!r}")
    batch_size = args.batch_size or config.get("batch_size") or 64
    device = resolve_device(args.device)

    # A trained checkpoint already contains every weight. Constructing the
    # equivalent uninitialized architecture avoids downloading TorchVision
    # weights when the recorded run used a pretrained starting point.
    construction_name = model_name.replace("_pretrained", "")
    model = create_model(construction_name, image_size=image_size).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("checkpoint must be produced by scripts/train.py and contain a 'model' state dictionary")
    model.load_state_dict(checkpoint["model"])

    print(f"checkpoint: {checkpoint_path}")
    print(f"config: {config_path}")
    print(f"model: {model_name}; image size: {image_size}; device: {device}")

    results = {}
    for data_path in args.data:
        loader = make_loader(
            data_path,
            image_size,
            batch_size,
            args.workers,
            training=False,
            pin_memory=device.type == "cuda",
        )
        metrics = evaluate(model, loader, device)
        metrics.pop("loss", None)
        results[data_path] = {"num_images": len(loader.dataset), **metrics}
        print(format_evaluation_metrics(data_path, metrics))

    run_dir = checkpoint_path.parent.parent
    output_path = resolve_output_path(args.output_json, run_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "model": model_name,
        "image_size": list(image_size),
        "device": str(device),
        "results": results,
    }
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print(f"metrics: {output_path}")


if __name__ == "__main__":
    main()
