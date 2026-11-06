"""Train the unpruned FP32 VGG16 baseline; quantization is a later stage.

See VGG16_BASELINE.md for dataset preparation, commands, and interpretation.
"""

import argparse
import json
import math
import os
from pathlib import Path
import random
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import datasets, transforms
from torchvision.models import vgg16


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("imagenet", "cifar100", "fake"), default="cifar100")
    parser.add_argument("--data-root", type=Path, default=Path("data/imagenet"))
    parser.add_argument("--download", action="store_true", help="Download CIFAR-100 only")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/vgg16-fp32"))
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=4, help="Physical batch size")
    parser.add_argument("--accumulation-steps", type=int, default=64,
                        help="Optimizer batch = batch size times accumulation steps")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-step-size", type=int, default=30)
    parser.add_argument("--lr-gamma", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--log-interval", type=int, default=100, help="Minibatches between progress reports")
    parser.add_argument("--resume", type=Path, help="Resume an epoch-boundary checkpoint")
    parser.add_argument("--evaluate", type=Path, help="Validate a saved checkpoint without training")
    args = parser.parse_args()
    for name in ("epochs", "batch_size", "accumulation_steps", "lr_step_size", "log_interval"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.workers < 0 or args.image_size < 32:
        parser.error("workers must be nonnegative and image size must be at least 32")
    values = (args.lr, args.momentum, args.weight_decay, args.lr_gamma)
    if not all(math.isfinite(x) for x in values):
        parser.error("optimizer and scheduler settings must be finite")
    if args.lr <= 0 or args.weight_decay < 0 or not 0 <= args.momentum < 1 or args.lr_gamma <= 0:
        parser.error("invalid optimizer or scheduler settings")
    if args.download and args.dataset != "cifar100":
        parser.error("only CIFAR-100 supports --download; prepare ImageNet separately")
    if args.resume and args.evaluate:
        parser.error("use either --resume or --evaluate")
    return args


def build_datasets(args):
    # Keep the original VGG classifier/pooling even for smaller input images.
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.image_size), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), normalize,
    ])
    val_transform = transforms.Compose([
        transforms.Resize(round(args.image_size * 256 / 224)),
        transforms.CenterCrop(args.image_size), transforms.ToTensor(), normalize,
    ])

    if args.dataset == "fake":
        # No network or real accuracy claims: a tiny integration check only.
        train = datasets.FakeData(4, (3, args.image_size, args.image_size), 10,
                                  transform=train_transform, random_offset=0)
        val = datasets.FakeData(4, (3, args.image_size, args.image_size), 10,
                                transform=val_transform, random_offset=100)
        return train, val, [str(i) for i in range(10)]
    if args.dataset == "cifar100":
        train = datasets.CIFAR100(args.data_root, train=True, download=args.download,
                                  transform=train_transform)
        val = datasets.CIFAR100(args.data_root, train=True, download=False, transform=val_transform)
        # Hold out training examples; leave the official test set untouched.
        indices = torch.randperm(len(train), generator=torch.Generator().manual_seed(args.seed)).tolist()
        return Subset(train, indices[5000:]), Subset(val, indices[:5000]), train.classes
    for split in ("train", "val"):
        if not (args.data_root / split).is_dir():
            raise ValueError(f"Missing {args.data_root / split}. Prepare ImageNet train/ and val/ "
                             "with one directory per class; see VGG16_BASELINE.md.")
    
    train = datasets.ImageFolder(args.data_root / "train", transform=train_transform)
    val = datasets.ImageFolder(args.data_root / "val", transform=val_transform)

    if train.class_to_idx != val.class_to_idx:
        raise ValueError("ImageNet train and val class mappings differ")
    if len(train.classes) != 1000:
        raise ValueError("The imagenet baseline requires all 1000 classes")
    return train, val, train.classes


def run_epoch(model, loader, device, optimizer=None, accumulation_steps=1, log_interval=100):
    training = optimizer is not None
    model.train(training)
    totals = torch.zeros(4, device=device, dtype=torch.float64)
    group_samples = 0
    step = 0
    try:
        total_steps = len(loader)
    except TypeError:  # Hugging Face streaming/other unsized iterable loaders.
        total_steps = None

    def finish_accumulation():
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(group_samples)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    if training:
        optimizer.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(training):
        for step, (images, targets) in enumerate(loader, 1):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, targets, reduction="sum")
            if not torch.isfinite(loss).item():
                raise RuntimeError("Nonfinite loss; reduce the learning rate and inspect the data")
            if training:
                loss.backward()
                group_samples += targets.size(0)
                if step % accumulation_steps == 0:
                    # Normalize actual samples, including partial final batches/groups.
                    finish_accumulation()
                    group_samples = 0
            with torch.no_grad():
                predictions = logits.topk(min(5, logits.size(1)), dim=1).indices
                correct = predictions.eq(targets[:, None])
                totals[0] += loss.detach()
                totals[1] += correct[:, 0].sum()
                totals[2] += correct.any(dim=1).sum()
                totals[3] += targets.size(0)
            if step % log_interval == 0 or step == total_steps:
                print(f"{'train' if training else 'val'} {step}/{total_steps or '?'} "
                      f"loss={totals[0].item() / totals[3].item():.4f}", flush=True)
    # Iterable loaders need not expose a length; flush the last partial group
    # when iteration ends rather than relying on a predicted last batch index.
    if training and group_samples:
        finish_accumulation()
    if step and total_steps is None and step % log_interval:
        print(f"{'train' if training else 'val'} {step}/end "
              f"loss={totals[0].item() / totals[3].item():.4f}", flush=True)
    loss_sum, top1, top5, count = totals.tolist()

    if count == 0:
        raise ValueError("Cannot evaluate an empty loader")
    
    return {"loss": loss_sum / count, "top1": 100 * top1 / count,
            "top5": 100 * top5 / count, "samples": int(count)}


def save_checkpoint(path, checkpoint):
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def main():
    args = parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    # VGG adaptive pooling has no deterministic CUDA backward kernel.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    device = torch.device(args.device)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}

    train, val, classes = build_datasets(args)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=device.type == "cuda", generator=generator)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                            pin_memory=device.type == "cuda", generator=torch.Generator().manual_seed(args.seed))
    model = vgg16(weights=None, num_classes=len(classes)).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, gamma=args.lr_gamma)

    start_epoch, best_top1, history = 0, -1.0, []
    checkpoint_path = args.resume or args.evaluate
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        keys = ["dataset", "image_size", "seed"]
        if args.resume:
            keys += ["batch_size", "accumulation_steps", "lr", "momentum", "weight_decay",
                     "lr_step_size", "lr_gamma", "workers"]
        for key in keys:
            if checkpoint["config"][key] != config[key]:
                raise ValueError(f"Checkpoint {key}={checkpoint['config'][key]!r} differs from command")
        if checkpoint["classes"] != classes:
            raise ValueError("Checkpoint class mapping differs from dataset")
        model.load_state_dict(checkpoint["model"])
        if args.resume:
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            start_epoch, best_top1 = checkpoint["epoch"], checkpoint["best_top1"]
            history = checkpoint["history"]
            random.setstate(checkpoint["python_rng"])
            torch.set_rng_state(checkpoint["torch_rng"])
            generator.set_state(checkpoint["loader_rng"])
            if device.type == "cuda" and checkpoint["cuda_rng"]:
                torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        del checkpoint
    print(f"VGG16 FP32 | {args.dataset} | {len(train)} train / {len(val)} val | {device}", flush=True)
    
    if args.dataset == "fake":
        print("SYNTHETIC SMOKE CHECK: accuracy is not a dataset baseline.", flush=True)
    if args.evaluate:
        print(json.dumps(run_epoch(model, val_loader, device, log_interval=args.log_interval)), flush=True)
        return
    if start_epoch >= args.epochs:
        raise ValueError("--epochs must exceed the number of completed checkpoint epochs")
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise ValueError("Use an empty output directory to preserve previous experiment artifacts")
    
    environment = {"torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
                   "cuda": torch.version.cuda,
                   "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)}
    (args.output_dir / "config.json").write_text(json.dumps({**config, "environment": environment}, indent=2))
    
    
    for epoch in range(start_epoch, args.epochs):
        start = time.monotonic()
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch + 1}/{args.epochs}, lr={lr:g}", flush=True)
        training = run_epoch(model, train_loader, device, optimizer, args.accumulation_steps, args.log_interval)
        validation = run_epoch(model, val_loader, device, log_interval=args.log_interval)
        scheduler.step()
        improved = validation["top1"] > best_top1
        best_top1 = max(best_top1, validation["top1"])
        record = {"epoch": epoch + 1, "lr": lr, "train": training, "val": validation,
                  "seconds": time.monotonic() - start}
        history.append(record)
        print(json.dumps(record), flush=True)
        (args.output_dir / "metrics.json").write_text(json.dumps(history, indent=2))
        checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "scheduler": scheduler.state_dict(), "epoch": epoch + 1, "best_top1": best_top1,
                      "config": config, "classes": classes, "history": history, "environment": environment,
                      "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
                      "loader_rng": generator.get_state(),
                      "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}
        save_checkpoint(args.output_dir / "last.pt", checkpoint)
        if improved:
            save_checkpoint(args.output_dir / "best.pt", checkpoint)
    
    print(f"Best validation top-1: {best_top1:.3f}%", flush=True)
    
    if (args.output_dir / "best.pt").exists():
        print(f"Best checkpoint in this segment: {args.output_dir / 'best.pt'}", flush=True)
    else:
        print("Best checkpoint belongs to an earlier training segment.", flush=True)


if __name__ == "__main__":
    main()
