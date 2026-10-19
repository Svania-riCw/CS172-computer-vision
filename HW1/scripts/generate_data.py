import argparse
import io
import json
import zipfile
from collections.abc import Iterator
from pathlib import Path

import tqdm
from PIL import Image

from cs172.generate_captcha import generate_captcha


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate labeled CAPTCHA images.")
    parser.add_argument("output_dir", type=Path, help="Output directory, or a .zip archive with --zip")
    parser.add_argument("-n", "--num-images", type=int, required=True, help="Number of images to generate")
    parser.add_argument("--zip", action="store_true", help="write directly to a ZIP archive instead of a directory")
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(260, 80),
        help="Image dimensions in pixels (default: 260 80)",
    )
    parser.add_argument("--num-digits", type=int, default=5, help="Digits per CAPTCHA (default: 5)")
    parser.add_argument(
        "--font-size-range",
        type=int,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(30, 40),
        help="Inclusive font-size range (default: 30 40)",
    )
    parser.add_argument(
        "--minimum-padding",
        type=int,
        nargs=2,
        metavar=("HORIZONTAL", "VERTICAL"),
        default=(5, 10),
        help="Minimum horizontal and vertical padding (default: 5 10)",
    )
    parser.add_argument(
        "--rotation-angle-range",
        type=int,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(-30, 30),
        help="Rotation range in degrees (default: -30 30)",
    )
    parser.add_argument("--no-rotation", action="store_true", help="Disable digit rotation")
    parser.add_argument("--max-noisy-lines", type=int, default=5, help="Maximum interference lines (default: 5)")
    parser.add_argument(
        "--noise-ratio",
        type=float,
        default=0.02,
        help="Fraction of pixels replaced by noise (default: 0.02)",
    )
    parser.add_argument("--border-width", type=int, default=2, help="Border width (default: 2)")
    parser.add_argument("--seed", type=int, default=0, help="Base seed for reproducible output")
    args = parser.parse_args()
    validate_args(parser, args)
    return args


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.num_images < 1:
        parser.error("--num-images must be at least 1")
    if args.num_digits < 1:
        parser.error("--num-digits must be at least 1")
    if any(dimension < 1 for dimension in args.image_size):
        parser.error("--image-size values must be positive")
    if args.font_size_range[0] < 1 or args.font_size_range[0] > args.font_size_range[1]:
        parser.error("--font-size-range must contain positive values in ascending order")
    if any(padding < 0 for padding in args.minimum_padding):
        parser.error("--minimum-padding values cannot be negative")
    if args.max_noisy_lines < 0:
        parser.error("--max-noisy-lines cannot be negative")
    if not 0 <= args.noise_ratio <= 1:
        parser.error("--noise-ratio must be between 0 and 1")
    if args.border_width < 0:
        parser.error("--border-width cannot be negative")
    if args.zip and args.output_dir.suffix.lower() != ".zip":
        parser.error("ZIP output path must end with .zip")


def generate_images(args: argparse.Namespace) -> Iterator[tuple[Image.Image, str]]:
    for index in tqdm.trange(args.num_images, desc="Generating images"):
        image, characters = generate_captcha(
            image_size=tuple(args.image_size),
            num_digits=args.num_digits,
            font_size_range=tuple(args.font_size_range),
            minimum_padding=tuple(args.minimum_padding),
            rotation_angle_range=None if args.no_rotation else tuple(args.rotation_angle_range),
            max_noisy_lines=args.max_noisy_lines,
            noise_ratio=args.noise_ratio,
            border_width=args.border_width,
            seed=args.seed + index,
        )
        yield image, "".join(characters)


def next_filename(characters: str, existing_names: set[str]) -> str:
    counter = 0
    filename = f"{characters}#{counter}.png"
    while filename in existing_names:
        counter += 1
        filename = f"{characters}#{counter}.png"
    existing_names.add(filename)
    return filename


def save_to_directory(args: argparse.Namespace, output_dir: Path) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"output path is not a directory: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    existing_names = {path.name for path in output_dir.iterdir() if path.is_file()}
    for image, characters in generate_images(args):
        filename = next_filename(characters, existing_names)
        image.save(output_dir / filename)


def save_to_zip(args: argparse.Namespace, output_path: Path) -> None:
    if output_path.exists() and not output_path.is_file():
        raise ValueError(f"output path is not a file: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, mode="a", compression=zipfile.ZIP_DEFLATED) as archive:
        existing_names = set(archive.namelist())
        for image, characters in generate_images(args):
            filename = next_filename(characters, existing_names)
            image_buffer = io.BytesIO()
            image.save(image_buffer, format="PNG")
            archive.writestr(filename, image_buffer.getvalue())


def save_manifest(args: argparse.Namespace, output_path: Path) -> None:
    manifest = vars(args).copy()
    manifest["output_dir"] = str(output_path)
    if args.zip:
        manifest_path = output_path.with_suffix(".json")
    else:
        manifest_path = output_path.parent / f"{output_path.name}.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
        file.write("\n")


def main() -> None:
    args = parse_args()
    output_path = args.output_dir.expanduser().resolve()

    # Directory output is the standard workflow; ZIP output is optional.
    if args.zip:
        save_to_zip(args, output_path)
    else:
        save_to_directory(args, output_path)
    save_manifest(args, output_path)

    destination = "ZIP archive" if args.zip else "directory"
    print(f"Generated {args.num_images} image(s) in {destination} {output_path}")


if __name__ == "__main__":
    main()
