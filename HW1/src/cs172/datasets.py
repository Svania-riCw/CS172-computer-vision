"""Datasets for five-digit CAPTCHA images."""

from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2


class CAPTCHA(Dataset):
    """Read labeled CAPTCHA images from a directory."""

    def __init__(
        self,
        source,
        target_size=None,
        image_size=None,
        transform: v2.Transform | None = None,
        cache_images: bool = False,
        num_images: int | None = None,
    ):
        self.source = os.fspath(source)
        self.image_size = target_size if target_size is not None else image_size
        self.transform = transform
        self.cache_images = cache_images
        self._cache: dict[str, torch.Tensor] = {}
        self.img_paths = sorted(self._find_images())
        if num_images is not None:
            self.img_paths = self.img_paths[:num_images]

    def _find_images(self):
        root = Path(self.source)
        if not root.is_dir():
            raise ValueError(f"image source is not a directory: {self.source}")
        return [
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]

    @staticmethod
    def _parse_label(filename) -> torch.Tensor:
        stem = Path(filename).stem
        digits = stem.split("#", maxsplit=1)[0]
        if len(digits) != 5 or not digits.isdecimal():
            raise ValueError(f"expected a five-digit label in filename: {filename}")
        return torch.tensor([int(digit) for digit in digits], dtype=torch.long)

    @staticmethod
    def _decode_bytes(data: bytes) -> torch.Tensor:
        with Image.open(io.BytesIO(data)) as image:
            rgb = image.convert("RGB")
            array = np.asarray(rgb, dtype=np.uint8).copy()
        return torch.from_numpy(array).permute(2, 0, 1)

    def _read_image(self, img_path) -> torch.Tensor:
        path = Path(img_path)
        return self._decode_bytes(path.read_bytes())

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        cache_key = os.fspath(img_path)
        if self.cache_images and cache_key in self._cache:
            image = self._cache[cache_key].clone()
        else:
            image = self._read_image(img_path)
            if self.cache_images:
                self._cache[cache_key] = image.clone()

        operations = []
        if self.image_size is not None:
            height, width = self.image_size
            operations.append(v2.Resize((height, width)))
        if self.transform is not None:
            operations.append(self.transform)
        operations.append(v2.ToDtype(torch.float32, scale=True))
        image = v2.Compose(operations)(image)
        return image, self._parse_label(img_path)


class ZIP_CAPTCHA(CAPTCHA):
    """Read labeled CAPTCHA images directly from a ZIP archive."""

    def _find_images(self):
        with zipfile.ZipFile(self.source) as archive:
            return sorted(
                name
                for name in archive.namelist()
                if Path(name).suffix.lower() in {".jpg", ".jpeg", ".png"}
            )

    def _read_image(self, img_path) -> torch.Tensor:
        with zipfile.ZipFile(self.source) as archive:
            return self._decode_bytes(archive.read(img_path))

