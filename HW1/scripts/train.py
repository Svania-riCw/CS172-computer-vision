"""Train a model to recognize all five digits in CAPTCHA images."""

import argparse
import json
import os
import warnings
from datetime import datetime
from uuid import uuid4

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms
from tqdm.auto import tqdm

from cs172.datasets import CAPTCHA, ZIP_CAPTCHA
from cs172.models import MODELS, create_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/training", help="directory or ZIP containing training images")
    parser.add_argument("--num-images", type=int, help="optional limit on number of training images to use")
    parser.add_argument("--val-data-dir", help="optional directory or ZIP containing validation images")
    parser.add_argument("--output-dir", default="outputs", help="experiment directory")
    parser.add_argument("--model", choices=MODELS, default="resnet18")
    parser.add_argument("--image-size", nargs=2, type=int, metavar=("HEIGHT", "WIDTH"), default=(224, 224))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augment", action="store_true", help="enable mild training augmentation")
    parser.add_argument(
        "--cache-images",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="cache decoded images (each worker keeps a separate cache)",
    )
    parser.add_argument("--tensorboard", action="store_true", help="enable TensorBoard logging")
    parser.add_argument("--resume", help="checkpoint from which to resume training")
    parser.add_argument("--weights-only", action="store_true", help="load only model weights from --resume")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_run_dir(args: argparse.Namespace) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{timestamp}-{uuid4().hex[:8]}"
    run_dir = os.path.abspath(os.path.join(args.output_dir, run_id))
    os.makedirs(run_dir, exist_ok=False)

    config = vars(args).copy()
    config["run_id"] = run_id
    config["run_dir"] = run_dir
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)
        file.write("\n")

    return run_dir


def make_loader(
    path: str,
    image_size: tuple[int, int],
    batch_size: int,
    workers: int,
    training: bool,
    cache_images: bool = False,
    augment: bool = False,
    num_images: int | None = None,
    pin_memory: bool = True,
) -> DataLoader:
    transform = None
    if training and augment:
        transform = transforms.Compose(
            [
                transforms.RandomAffine(degrees=5, translate=(0.05, 0.02), scale=(0.95, 1.05)),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            ]
        )

    # Each worker has its own dataset and cache, so caching with multiple
    # workers can multiply memory use.
    dataset_class = ZIP_CAPTCHA if path.lower().endswith(".zip") else CAPTCHA
    dataset = dataset_class(
        path,
        target_size=image_size,
        transform=transform,
        cache_images=cache_images,
        num_images=num_images,
    )
    if not dataset:
        raise ValueError(f"no images found in {path!r}")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
        drop_last=training,
    )


def train_epoch(model, loader, optimizer, device, writer, epoch: int) -> dict[str, float]:
    model.train()
    loss_sum = 0.0
    correct_digits = 0
    correct_images = 0
    num_images = 0
    num_digits = 0

    progress = tqdm(loader, desc="train", leave=False)
    for batch_index, (images, targets) in enumerate(progress):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss, _ = model.compute_loss(logits, targets)
        loss.backward()
        optimizer.step()

        predictions = logits.argmax(dim=1)
        correct = predictions.eq(targets)
        batch_size = targets.shape[0]
        loss_sum += loss.item() * batch_size
        correct_digits += correct.sum().item()
        correct_images += correct.all(dim=1).sum().item()
        num_images += batch_size
        num_digits += targets.numel()

        step = epoch * len(loader) + batch_index
        if writer is not None:
            writer.add_scalar("train/batch_loss", loss.item(), step)
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return {
        "loss": loss_sum / num_images,
        "digit_acc": correct_digits / num_digits,
        "image_acc": correct_images / num_images,
    }


@torch.inference_mode()
def evaluate(model, loader, device) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    correct_digits = 0
    correct_images = 0
    num_images = 0
    num_digits = 0

    for images, targets in tqdm(loader, desc="val", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss, _ = model.compute_loss(logits, targets)
        correct = logits.argmax(dim=1).eq(targets)
        batch_size = targets.shape[0]

        loss_sum += loss.item() * batch_size
        correct_digits += correct.sum().item()
        correct_images += correct.all(dim=1).sum().item()
        num_images += batch_size
        num_digits += targets.numel()

    return {
        "loss": loss_sum / num_images,
        "digit_acc": correct_digits / num_digits,
        "image_acc": correct_images / num_images,
    }


def format_metrics(name: str, metrics: dict[str, float]) -> str:
    return f"{name} loss {metrics['loss']:.4f}, " f"digit {metrics['digit_acc']:.2%}, image {metrics['image_acc']:.2%}"


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_acc) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_image_acc": best_acc,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; this training script requires a GPU")
    if any(size < 1 for size in args.image_size):
        raise ValueError("--image-size values must be positive")
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if args.weights_only and not args.resume:
        raise ValueError("--weights-only requires --resume")

    set_seed(args.seed)
    device = torch.device("cuda")
    print(f"device: {device}")
    run_dir = create_run_dir(args)
    print(f"run: {run_dir}")
    image_size = tuple(args.image_size)

    train_loader = make_loader(
        args.data_dir,
        image_size,
        args.batch_size,
        args.workers,
        training=True,
        cache_images=args.cache_images,
        augment=args.augment,
        num_images=args.num_images,
    )
    val_loader = None
    if args.val_data_dir:
        val_loader = make_loader(
            args.val_data_dir,
            image_size,
            args.batch_size,
            args.workers,
            training=False,
            cache_images=args.cache_images,
        )

    print(f"training images: {len(train_loader.dataset)}")
    if val_loader is not None:
        print(f"validation images: {len(val_loader.dataset)}")

    model = create_model(args.model, image_size=image_size).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    best_acc = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        if not args.weights_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            start_epoch = checkpoint["epoch"] + 1
            best_acc = checkpoint.get("best_image_acc")
            checkpoint_epochs = checkpoint["scheduler"].get("T_max")
            if checkpoint_epochs != args.epochs:
                warnings.warn(
                    f"--epochs is {args.epochs}, but the checkpoint was configured for "
                    f"{checkpoint_epochs}; the checkpoint scheduler will be resumed unchanged",
                    stacklevel=2,
                )
            if start_epoch >= args.epochs:
                raise ValueError(
                    f"checkpoint has completed {start_epoch} epochs; "
                    f"--epochs must be greater than {start_epoch} to resume"
                )

    writer = None
    if args.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as error:
            raise RuntimeError("install the optional 'tensorboard' dependency to use --tensorboard") from error
        tensorboard_dir = os.path.join(run_dir, "tensorboard")
        os.makedirs(tensorboard_dir)
        writer = SummaryWriter(log_dir=tensorboard_dir)

    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    try:
        for epoch in range(start_epoch, args.epochs):
            train_metrics = train_epoch(model, train_loader, optimizer, device, writer, epoch)
            val_metrics = evaluate(model, val_loader, device) if val_loader is not None else None

            learning_rate = optimizer.param_groups[0]["lr"]
            scheduler.step()

            if writer is not None:
                metrics_by_split = {"train": train_metrics, "val": val_metrics}
                for split, metrics in metrics_by_split.items():
                    if metrics is not None:
                        for name, value in metrics.items():
                            writer.add_scalar(f"{split}/{name}", value, epoch + 1)
                writer.add_scalar("train/learning_rate", learning_rate, epoch + 1)

            summary = f"epoch {epoch + 1:02d}/{args.epochs:02d} | {format_metrics('train', train_metrics)}"
            if val_metrics is not None:
                summary += f" | {format_metrics('val', val_metrics)}"
            print(summary)

            is_best = val_metrics is not None and (best_acc is None or val_metrics["image_acc"] > best_acc)
            if is_best:
                best_acc = val_metrics["image_acc"]

            os.makedirs(checkpoint_dir, exist_ok=True)
            latest_path = os.path.join(checkpoint_dir, "latest.pt")
            save_checkpoint(latest_path, model, optimizer, scheduler, epoch, best_acc)
            if is_best:
                best_path = os.path.join(checkpoint_dir, "best.pt")
                save_checkpoint(best_path, model, optimizer, scheduler, epoch, best_acc)
    except KeyboardInterrupt:
        print("training interrupted by user")
    finally:
        if writer is not None:
            writer.close()

    if best_acc is not None:
        print(f"best validation image acc: {best_acc:.2%}")
    if os.path.isdir(checkpoint_dir):
        print(f"checkpoints: {checkpoint_dir}")


if __name__ == "__main__":
    main()
