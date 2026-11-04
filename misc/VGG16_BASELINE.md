# VGG16 FP32 baseline

`vgg16_model_quant_and_prune.py` currently implements the baseline stage: train
`torchvision.models.vgg16(weights=None)` from scratch, with the original
convolution/pooling/classifier architecture and no batch normalization. No
pretrained parameters, quantization, pruning, AMP, or TF32 are used.

Run commands from the repository root using the existing `.venv/bin/python`.
The script requires compatible PyTorch and torchvision installations.

## ImageNet-1K (primary experiment)

Download ILSVRC2012 separately under its access terms. Torchvision cannot fetch
ImageNet automatically. Prepare both splits in ImageFolder layout:

```text
data/imagenet/
  train/
    n01440764/*.JPEG
    ... (1000 class directories)
  val/
    n01440764/*.JPEG
    ... (the same 1000 class directories)
```

The official validation archive contains flat images; organize them using the
official devkit labels, not arbitrary filenames or guessed labels. The script
checks that the two class-to-index mappings agree. It expects extracted data,
not tar archives. Record the dataset version/source alongside the experiment;
1000 folders alone do not prove it is the complete official dataset.

```bash
.venv/bin/python misc/vgg16_model_quant_and_prune.py \
  --dataset imagenet --data-root data/imagenet \
  --output-dir runs/vgg16-imagenet-fp32
```

Defaults: 90 epochs, SGD with learning rate 0.01, momentum 0.9, weight decay
0.0001, and learning-rate reduction by 0.1 after epochs 30 and 60. Input images
use random resized crops and horizontal flips for training. Validation uses
resize to 256 and center crop to 224. Both use ImageNet mean/std normalization.

The physical batch size is 4, with 64 accumulation steps (effective batch 256,
except the final group). Accumulation averages gradients by the actual sample
count. This accommodates smaller GPUs; reduce the physical batch and increase
accumulation if needed. On a larger GPU, for example, use `--batch-size 32
--accumulation-steps 8`. VGG16 has dropout but no batch normalization, so gradient
accumulation is suitable, though it does not promise identical numerical results
to a single large batch. The recipe is based on torchvision's reference settings;
this script uses one device and has not reproduced the published result.

Torchvision lists **71.592% top-1 / 90.382% top-5** for its published VGG16 weights
on ImageNet-1K. These are reference values, **not measured results from this
project** and not a guarantee for this run. Establish our measured baseline from
the validation metrics after training.

## Automatically downloadable alternative: CIFAR-100

```bash
.venv/bin/python misc/vgg16_model_quant_and_prune.py \
  --dataset cifar100 --data-root data/cifar100 --download \
  --image-size 32 --output-dir runs/vgg16-cifar100-fp32
```

This uses a seeded 45,000/5,000 split of the official training set. Training and
validation have separate transforms; the official test set is left untouched.
It keeps VGG16's adaptive 7x7 pooling and large classifier, with 100 outputs.
At 32px this pools a 1x1 feature map to 7x7: it is a convenient smaller experiment,
not a CIFAR-optimized VGG architecture. Set `--image-size 224` to use ImageNet-sized
inputs at greater compute cost. ImageNet normalization is retained deliberately.
The ImageNet hyperparameters are a starting point, not a tuned CIFAR recipe.
CIFAR-100 accuracy cannot be substituted for ImageNet accuracy. Compare all
quantization/pruning variants with the same dataset, split, seed, resolution,
preprocessing, architecture, and evaluation protocol as their FP32 baseline.

## Outputs and continuation

Each epoch prints and saves sample-weighted cross-entropy, top-1, and top-5
accuracy (percentages) for training and validation. Training metrics include
augmentation/dropout and describe predictions while weights are changing;
validation metrics are the baseline comparison. Outputs include:

- `config.json`: command settings and PyTorch/torchvision/CUDA/device versions.
- `metrics.json`: all completed epochs and elapsed epoch times.
- `best.pt`: checkpoint selected by validation top-1.
- `last.pt`: latest completed epoch, including optimizer/scheduler and RNG states.

Checkpoints contain class order, run configuration, metric history, and model
state for later compression experiments. Each is about 1.1 GB for ImageNet VGG16
with momentum state; allow additional space for atomic checkpoint replacement.
Use an empty output directory for every invocation, including resumed segments,
to preserve previous results. Resume from a completed epoch, repeating the
original dataset, seed, image size, optimizer, batch, and worker settings:

```bash
.venv/bin/python misc/vgg16_model_quant_and_prune.py \
  --dataset imagenet --data-root data/imagenet \
  --resume runs/vgg16-imagenet-fp32/last.pt --epochs 90 \
  --output-dir runs/vgg16-imagenet-fp32-resumed

.venv/bin/python misc/vgg16_model_quant_and_prune.py \
  --dataset imagenet --data-root data/imagenet \
  --evaluate runs/vgg16-imagenet-fp32/best.pt
```

Resumed histories include previous epochs. If a resumed segment never improves
the best validation accuracy, its `best.pt` is not written; use the prior best
checkpoint. Dataset contents must remain the same when resuming. Seeds and RNG
states support repeatability on a fixed environment. CUDA adaptive-pooling
backward can be nondeterministic, so deterministic mode warns instead of aborting;
bitwise reproducibility across runs/hardware/software is not guaranteed.

## Pipeline check without downloading data

```bash
.venv/bin/python misc/vgg16_model_quant_and_prune.py \
  --dataset fake --epochs 1 --image-size 32 --batch-size 1 \
  --accumulation-steps 3 --workers 0 --output-dir /tmp/vgg16-smoke
```

This runs the actual VGG16 on four synthetic training and four synthetic
validation images, exercising backward, an incomplete accumulation group,
validation, and checkpoint writing. Its accuracy is not a model baseline.

No completed ImageNet or CIFAR-100 training result is recorded yet. ImageNet data
is not present in this repository. The available local GPU is an NVIDIA T400
with 4 GB VRAM; a full ImageNet run is a substantial compute job even when it fits
in memory. Do not infer training accuracy from a smoke check.

## Sources

- [Torchvision VGG16 and published accuracy](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.vgg16.html)
- [Torchvision classification recipe (AlexNet and VGG)](https://github.com/pytorch/vision/tree/main/references/classification#alexnet-and-vgg)
- [ImageNet dataset requirements](https://docs.pytorch.org/vision/main/_modules/torchvision/datasets/imagenet.html)
- [CIFAR-100 dataset API](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR100.html)

## Local verification (November 3, 2026)

All 20 project unit tests passed, including four new baseline checks. Actual
VGG16 CUDA smoke runs passed training, validation, checkpoint saving, resumption,
and checkpoint-only evaluation. A physical batch of four 224x224 images also
completed on the local T400. These checks used synthetic data only; temporary
model checkpoints were removed after verification.
