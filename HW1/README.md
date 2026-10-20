# HW1 - CAPTCHA Recognition

This homework builds an end-to-end computer-vision pipeline for recognizing five-digit numeric CAPTCHAs. It includes CAPTCHA image generation, RGB image loading and preprocessing, ResNet model design, GPU training, checkpoint-based evaluation, and controlled experiments on data size, augmentation, regularization, batch size, learning rate, backbone depth, and spatial pooling.

See [`INSTRUCTION.md`](INSTRUCTION.md) for the complete requirements and [`main.ipynb`](main.ipynb) for the experiment report.

## Environment and reproduction

The work was run in the `cs172` Conda environment with PyTorch 2.5.1 and CUDA on
an NVIDIA GeForce RTX 4060 Laptop GPU:

```bash
conda activate cs172
pip install -e .
PYTHONPATH=src python scripts/generate_data.py data/training_mixed/clean -n 2000 --seed 1000 --no-rotation --max-noisy-lines 0 --noise-ratio 0 --digit-color-range 0 0
# Open main.ipynb for the fixed 80/20 split, training curves, and checkpoint selection.
PYTHONPATH=src python scripts/evaluate.py --checkpoint outputs/resnet18_best.pt --model resnet18 --data data/test/test0 data/test/test1 data/test/test2 data/test/test3 data/test/test4 --image-size 80 260
```

The selected notebook experiment uses the original 2,000-image training set with
a fixed 80/20 split. Its best validation checkpoint reached 96.25% full-image
accuracy. On the required 500 test images it achieved 64.12% per-digit accuracy
and 14.20% full-image accuracy.
The assignment score is 0.00 because both thresholds in the supplied score
formula are strict; the complete metrics and per-test-set breakdown are printed
in `main.ipynb`.

Generated datasets and checkpoints are intentionally ignored by Git. The source,
notebook, README, and configuration files are the reproducible submission
artifacts.
