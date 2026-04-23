from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

POSITIONS = ("bottom-left", "bottom-right", "top-left", "top-right", "center")


def load_font(path: str | None, size: int) -> ImageFont.ImageFont:
    if path:
        p = Path(path)
        if p.exists():
            return ImageFont.truetype(str(p), size=size)
        print(f"[text_overlay] warn: font not found at {p}, using PIL default")
    return ImageFont.load_default()


def anchor_for(position: str, padding: int, img_w: int, img_h: int,
               block_w: int, block_h: int) -> tuple[int, int]:
    if position == "bottom-left":
        return padding, img_h - block_h - padding
    if position == "bottom-right":
        return img_w - block_w - padding, img_h - block_h - padding
    if position == "top-left":
        return padding, padding
    if position == "top-right":
        return img_w - block_w - padding, padding
    if position == "center":
        return (img_w - block_w) // 2, (img_h - block_h) // 2
    raise ValueError(f"unknown position: {position}")


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--brand", required=True)
    p.add_argument("--tagline", required=True)
    p.add_argument("--font", default=None, help="Path to .ttf / .otf font.")
    p.add_argument("--brand-size", type=int, default=64)
    p.add_argument("--tagline-size", type=int, default=32)
    p.add_argument("--position", choices=POSITIONS, default="bottom-left")
    p.add_argument("--padding", type=int, default=48)
    p.add_argument("--color", default="#FFFFFF")
    p.add_argument("--shadow", action="store_true", help="Draw a soft shadow under the text.")
    p.add_argument("--gap", type=int, default=12, help="Vertical gap between brand and tagline.")
    args = p.parse_args()

    img = Image.open(args.input).convert("RGB")
    draw = ImageDraw.Draw(img)

    brand_font = load_font(args.font, args.brand_size)
    tagline_font = load_font(args.font, args.tagline_size)

    bw, bh = text_size(draw, args.brand, brand_font)
    tw, th = text_size(draw, args.tagline, tagline_font)
    block_w = max(bw, tw)
    block_h = bh + args.gap + th

    x, y = anchor_for(args.position, args.padding, img.width, img.height, block_w, block_h)

    def draw_line(text: str, font: ImageFont.ImageFont, xx: int, yy: int) -> None:
        if args.shadow:
            draw.text((xx + 2, yy + 2), text, font=font, fill=(0, 0, 0, 128))
        draw.text((xx, yy), text, font=font, fill=args.color)

    draw_line(args.brand, brand_font, x, y)
    draw_line(args.tagline, tagline_font, x, y + bh + args.gap)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    img.save(args.output, format="PNG", optimize=True)
    print(f"[text_overlay] wrote {args.output}")


if __name__ == "__main__":
    main()
