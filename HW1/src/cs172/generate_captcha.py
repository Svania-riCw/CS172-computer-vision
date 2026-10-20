"""Create synthetic CAPTCHA images.

The generator builds one image in small stages: draw digits, optionally rotate
each digit, then add the CAPTCHA interference effects and border.
"""

import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a font that is available on this Linux environment."""
    candidates = (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
    )
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def generate_captcha(
    image_size=(260, 80),
    num_digits=5,
    font_size_range=(30, 40),
    minimum_padding=(5, 10),
    rotation_angle_range=(-30, 30),
    max_noisy_lines=5,
    noise_ratio=0.02,
    border_width=2,
    digit_color_range=(0, 160),
    position_jitter=0,
    free_positioning=False,
    seed=None,
):
    """Return one CAPTCHA image and its list of digit characters.

    The random generator is local to this call, so passing the same seed
    recreates the same label and image.
    """
    width, height = image_size
    padding_x, padding_y = minimum_padding
    rng = random.Random(seed)

    label = [str(rng.randrange(10)) for _ in range(num_digits)]
    font_size = rng.randint(font_size_range[0], font_size_range[1])
    font = _load_font(font_size)

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    cell_width = (width - 2 * padding_x) // num_digits

    free_x_positions = None
    if free_positioning:
        text_widths = [draw.textbbox((0, 0), digit, font=font)[2] for digit in label]
        available_gap = max(0, width - 2 * padding_x - sum(text_widths))
        gaps = [0] * (num_digits + 1)
        for _ in range(available_gap):
            gaps[rng.randrange(len(gaps))] += 1
        free_x_positions = []
        cursor = padding_x + gaps[0]
        for index, text_width in enumerate(text_widths):
            free_x_positions.append(cursor)
            cursor += text_width + gaps[index + 1]

    for index, digit in enumerate(label):
        left, top, right, bottom = draw.textbbox((0, 0), digit, font=font)
        text_width = right - left
        text_height = bottom - top
        available_x = max(0, cell_width - text_width)
        available_y = max(0, height - 2 * padding_y - text_height)

        if free_x_positions is not None:
            x = free_x_positions[index]
        else:
            jitter = rng.randint(-position_jitter, position_jitter)
            x = padding_x + index * cell_width + rng.randint(0, available_x) + jitter
            x = max(padding_x + index * cell_width, x)
            x = min(padding_x + (index + 1) * cell_width - text_width, x)
        y = padding_y + rng.randint(0, available_y)
        color = tuple(rng.randint(digit_color_range[0], digit_color_range[1]) for _ in range(3))

        if rotation_angle_range is None:
            draw.text((x - left, y - top), digit, fill=color, font=font)
            continue

        # Draw on a transparent layer so rotation does not rotate the whole
        # CAPTCHA or overwrite the white background.
        layer = Image.new("RGBA", (text_width, text_height), (0, 0, 0, 0))
        layer_draw = ImageDraw.Draw(layer)
        layer_draw.text((-left, -top), digit, fill=(*color, 255), font=font)
        angle = rng.randint(rotation_angle_range[0], rotation_angle_range[1])
        rotated = layer.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)

        max_x = max(0, cell_width - rotated.width)
        max_y = max(0, height - 2 * padding_y - rotated.height)
        if free_x_positions is not None:
            # Keep the rotated digit aligned with the same free horizontal
            # slot used for the unrotated branch, while allowing for the
            # extra width introduced by ``expand=True``.
            paste_x = free_x_positions[index] + rng.randint(-rotated.width // 8, rotated.width // 8)
            paste_x = max(padding_x, min(width - rotated.width - padding_x, paste_x))
        else:
            paste_x = padding_x + index * cell_width + rng.randint(0, max_x)
        paste_y = padding_y + rng.randint(0, max_y)
        image.paste(rotated, (paste_x, paste_y), rotated)

    # Add a small number of lines over the digits to simulate interference.
    for _ in range(rng.randint(0, max_noisy_lines)):
        line_color = tuple(rng.randint(0, 180) for _ in range(3))
        start = (rng.randint(0, width - 1), rng.randint(0, height - 1))
        end = (rng.randint(0, width - 1), rng.randint(0, height - 1))
        draw.line((start, end), fill=line_color, width=1)

    # Replace a fixed fraction of pixels with random RGB values. Using a
    # separate coordinate list keeps the requested ratio easy to verify.
    noisy_pixels = int(round(width * height * noise_ratio))
    for _ in range(noisy_pixels):
        point = (rng.randrange(width), rng.randrange(height))
        pixel_color = tuple(rng.randrange(256) for _ in range(3))
        image.putpixel(point, pixel_color)

    if border_width:
        draw.rectangle(
            (0, 0, width - 1, height - 1),
            outline=(0, 0, 0),
            width=border_width,
        )

    return image, label

