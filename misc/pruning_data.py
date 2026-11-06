"""Hugging Face ImageNet streaming input for the pruning notebook."""

from dataclasses import dataclass

import torch
from torchvision import transforms
from torchvision.models import VGG16_Weights


@dataclass
class ImageNetCollator:
    """Turn HF image/label records into the tensor pair expected by run_epoch."""

    transform: object
    num_classes: int = 1000

    def __call__(self, records):
        labels = torch.tensor([int(record["label"]) for record in records], dtype=torch.long)
        if ((labels < 0) | (labels >= self.num_classes)).any():
            raise ValueError("ImageNet evaluation/training requires labeled train or validation data")
        images = torch.stack([self.transform(record["image"].convert("RGB")) for record in records])
        return images, labels


def build_hf_imagenet_data(config, include_training=True):
    """Return repeatable finite HF streams and their train/validation collators.

    Selection seed and buffer stay fixed across epochs/trials. Augmentation is
    resampled by the loader. Keep worker count fixed when comparing ratios.
    None sample limits use the complete respective split; no test data is read.
    """
    from datasets import load_dataset

    for key in ("train_samples", "validation_samples"):
        limit = config.get(key)
        if limit is not None and (not isinstance(limit, int) or limit < 1):
            raise ValueError(f"{key} must be a positive integer or None")
    buffer_size = config.get("shuffle_buffer", 10000)
    if not isinstance(buffer_size, int) or buffer_size < 1:
        raise ValueError("shuffle_buffer must be a positive integer")
    weights = VGG16_Weights[config["weights"]]
    validation_transform = weights.transforms()
    if config["image_size"] != validation_transform.crop_size[0]:
        raise ValueError("Use the pretrained weights' evaluation crop size")

    def load_split(split, limit, shuffle):
        stream = load_dataset(config["dataset_id"], split=split, streaming=True)
        if shuffle:
            stream = stream.shuffle(seed=config["seed"], buffer_size=buffer_size)
        if limit is not None:
            stream = stream.take(limit)
        return stream

    validation = load_split("validation", config.get("validation_samples"),
                            shuffle=config.get("validation_samples") is not None)
    classes = validation.features["label"].names
    if len(classes) != len(weights.meta["categories"]):
        raise ValueError("Expected the 1000-class ImageNet-1K label space")
    training = None
    if include_training:
        training = load_split("train", config.get("train_samples"), shuffle=True)
        if training.features["label"].names != classes:
            raise ValueError("Training and validation label mappings differ")
    training_transform = transforms.Compose([
        transforms.RandomResizedCrop(config["image_size"]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(validation_transform.mean, validation_transform.std),
    ])
    return (training, validation, classes,
            ImageNetCollator(training_transform), ImageNetCollator(validation_transform))
