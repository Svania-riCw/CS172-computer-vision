# !/bin/bash

# Default
python scripts/train.py \
  --output-dir outputs/captcha100k-resnet18-aug \
  --data-dir data/training100k.zip \
  --val-data-dir data/validation.zip \
  --model resnet18 \
  --epochs 20 \
  --batch-size 64 \
  --learning-rate 0.001 \
  --weight-decay 0.0001 \
  --augment \
  --tensorboard

# 20k training images
python scripts/train.py \
  --output-dir outputs/captcha20k-resnet18-aug \
  --num-images 20000 \
  --data-dir data/training100k.zip \
  --val-data-dir data/validation.zip \
  --model resnet18 \
  --epochs 20 \
  --batch-size 64 \
  --learning-rate 0.001 \
  --weight-decay 0.0001 \
  --augment \
  --tensorboard

# 10k training images
python scripts/train.py \
  --output-dir outputs/captcha10k-resnet18-aug \
  --num-images 10000 \
  --data-dir data/training100k.zip \
  --val-data-dir data/validation.zip \
  --model resnet18 \
  --epochs 20 \
  --batch-size 64 \
  --learning-rate 0.001 \
  --weight-decay 0.0001 \
  --augment \
  --tensorboard

# 10k training images without augmentation
python scripts/train.py \
  --output-dir outputs/captcha10k-resnet18 \
  --num-images 10000 \
  --data-dir data/training100k.zip \
  --val-data-dir data/validation.zip \
  --model resnet18 \
  --epochs 20 \
  --batch-size 64 \
  --learning-rate 0.001 \
  --weight-decay 0.0001 \
  --tensorboard

# 10k training images without augmentation and without weight decay
python scripts/train.py \
  --output-dir outputs/captcha10k-resnet18-no-aug-no-wd \
  --num-images 10000 \
  --data-dir data/training100k.zip \
  --val-data-dir data/validation.zip \
  --model resnet18 \
  --epochs 20 \
  --batch-size 64 \
  --learning-rate 0.001 \
  --weight-decay 0.0 \
  --tensorboard

# 10k training images with augmentation and different batch size
python scripts/train.py \
  --output-dir outputs/captcha10k-resnet18-aug-batch32 \
  --num-images 10000 \
  --data-dir data/training100k.zip \
  --val-data-dir data/validation.zip \
  --model resnet18 \
  --epochs 20 \
  --batch-size 32 \
  --learning-rate 0.001 \
  --weight-decay 0.0001 \
  --augment \
  --tensorboard

# 10k training images with augmentation and lower learning rate
python scripts/train.py \
  --output-dir outputs/captcha10k-resnet18-aug-lr1e_4 \
  --num-images 10000 \
  --data-dir data/training100k.zip \
  --val-data-dir data/validation.zip \
  --model resnet18 \
  --epochs 20 \
  --batch-size 64 \
  --learning-rate 0.0001 \
  --weight-decay 0.0001 \
  --augment \
  --tensorboard

# resnet34 with 10k training images and augmentation
python scripts/train.py \
  --output-dir outputs/captcha10k-resnet34-aug \
  --num-images 10000 \
  --data-dir data/training100k.zip \
  --val-data-dir data/validation.zip \
  --model resnet34 \
  --epochs 20 \
  --batch-size 64 \
  --learning-rate 0.001 \
  --weight-decay 0.0001 \
  --augment \
  --tensorboard