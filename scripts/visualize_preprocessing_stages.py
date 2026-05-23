"""Create a paper-ready panel showing HarmonIt preprocessing stages."""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_dilation, binary_fill_holes, label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-path", default="data/abide_manifest.csv")
    parser.add_argument("--subject-id", default="USM_50514")
    parser.add_argument("--slice-idx", type=int, default=253)
    parser.add_argument("--out", default="outputs/figures/preprocessing_stages_usm_50514.png")
    parser.add_argument("--format", choices=("png", "pdf", "both"), default="both")
    parser.add_argument("--out-hw", nargs=2, type=int, default=(256, 256))
    parser.add_argument("--head-mask-thr", type=float, default=0.08)
    parser.add_argument("--head-mask-dilate", type=int, default=3)
    parser.add_argument("--fg-bbox-margin", type=int, default=20)
    return parser.parse_args()


def robust_normalize(vol: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    v = vol.astype(np.float32, copy=True)
    bg = v == 0
    fg = ~bg
    if fg.any():
        vals = v[fg]
        p1, p99 = np.percentile(vals, [1, 99])
        if p99 <= p1 + eps:
            p1 = float(vals.min())
            p99 = float(vals.max())
        v_fg = np.clip(vals, p1, p99)
        v[fg] = np.clip((v_fg - p1) / ((p99 - p1) + eps), 0.0, 1.0)
    v[bg] = 0.0
    return v


def make_head_mask(img2d: np.ndarray, thr: float, dilate_iters: int) -> np.ndarray:
    mask = np.abs(img2d) > thr
    if not mask.any():
        return np.ones_like(img2d, dtype=bool)
    components, n_components = label(mask)
    if n_components > 0:
        counts = np.bincount(components.ravel())
        counts[0] = 0
        mask = components == int(np.argmax(counts))
    mask = binary_fill_holes(mask)
    if dilate_iters > 0:
        mask = binary_dilation(mask, iterations=dilate_iters)
    return mask.astype(bool)


def bbox_from_mask(mask: np.ndarray, margin: int) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(ys) == 0 or len(xs) == 0:
        h, w = mask.shape
        return 0, h, 0, w
    y0 = max(0, int(ys.min()) - margin)
    y1 = min(mask.shape[0], int(ys.max()) + margin + 1)
    x0 = max(0, int(xs.min()) - margin)
    x1 = min(mask.shape[1], int(xs.max()) + margin + 1)
    return y0, y1, x0, x1


def crop_with_bbox(img2d: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    y0, y1, x0, x1 = bbox
    return img2d[y0:y1, x0:x1]


def resize_to_hw(img2d: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    img = Image.fromarray(np.asarray(img2d, dtype=np.float32), mode="F")
    return np.asarray(img.resize((out_hw[1], out_hw[0]), Image.Resampling.BILINEAR), dtype=np.float32)


def to_uint8(image: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, [p_low, p_high])
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    arr = np.clip((arr - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    return (arr * 255.0).astype(np.uint8)


def image_from_array(image: np.ndarray, size: tuple[int, int] = (256, 256)) -> Image.Image:
    return Image.fromarray(to_uint8(image), mode="L").convert("RGB").resize(size, Image.Resampling.BICUBIC)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/urw-base35/NimbusSans-Bold.otf" if bold else "/usr/share/fonts/urw-base35/NimbusSans-Regular.otf",
        "/usr/share/fonts/google-droid/DroidSans-Bold.ttf" if bold else "/usr/share/fonts/google-droid/DroidSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def overlay_mask(base: np.ndarray, mask: np.ndarray) -> Image.Image:
    img = image_from_array(base)
    rgba = img.convert("RGBA")
    overlay = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
    resized_mask = Image.fromarray(mask.astype(np.uint8) * 255, mode="L").resize(rgba.size, Image.Resampling.NEAREST)
    color = Image.new("RGBA", rgba.size, (42, 157, 143, 90))
    overlay.paste(color, mask=resized_mask)
    return Image.alpha_composite(rgba, overlay).convert("RGB")


def draw_bbox(base: np.ndarray, bbox: tuple[int, int, int, int]) -> Image.Image:
    img = image_from_array(base)
    h, w = base.shape
    scale_x = img.width / float(w)
    scale_y = img.height / float(h)
    y0, y1, x0, x1 = bbox
    box = [x0 * scale_x, y0 * scale_y, (x1 - 1) * scale_x, (y1 - 1) * scale_y]
    draw = ImageDraw.Draw(img)
    for offset in range(3):
        draw.rectangle(
            [box[0] - offset, box[1] - offset, box[2] + offset, box[3] + offset],
            outline=(230, 111, 81),
        )
    return img


def save_panel(panels: list[tuple[str, Image.Image]], out_path: Path, title: str, save_pdf: bool) -> None:
    cols = 3
    rows = int(np.ceil(len(panels) / cols))
    panel_size = 360
    cell_w = 430
    cell_h = 455
    margin_x = 44
    top = 105
    canvas = Image.new("RGB", (margin_x * 2 + cols * cell_w, top + rows * cell_h + 28), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = font(38)
    label_font = font(28)

    bbox = draw.textbbox((0, 0), title, font=title_font)
    draw.text(((canvas.width - (bbox[2] - bbox[0])) / 2, 24), title, fill=(20, 20, 20), font=title_font)

    for i, (label, img) in enumerate(panels):
        row, col = divmod(i, cols)
        x0 = margin_x + col * cell_w + (cell_w - panel_size) // 2
        y0 = top + row * cell_h + 46
        label_bbox = draw.textbbox((0, 0), label, font=label_font)
        draw.text(
            (margin_x + col * cell_w + (cell_w - (label_bbox[2] - label_bbox[0])) / 2, top + row * cell_h),
            label,
            fill=(20, 20, 20),
            font=label_font,
        )
        canvas.paste(img.resize((panel_size, panel_size), Image.Resampling.BICUBIC), (x0, y0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, dpi=(300, 300))
    if save_pdf:
        canvas.save(out_path.with_suffix(".pdf"), resolution=300.0)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    manifest = pd.read_csv(args.manifest_path)
    row = manifest.loc[manifest["subject_id"].astype(str) == args.subject_id]
    if row.empty:
        raise ValueError(f"Subject not found in manifest: {args.subject_id}")

    t1_path = Path(row.iloc[0]["t1_path"])
    if not t1_path.exists():
        parts = t1_path.parts
        if "data" in parts:
            t1_path = repo_root.joinpath(*parts[parts.index("data") :])
    if not t1_path.exists():
        raise FileNotFoundError(f"Could not resolve T1 path for {args.subject_id}: {row.iloc[0]['t1_path']}")
    image = nib.as_closest_canonical(nib.load(str(t1_path)))
    raw = image.get_fdata(dtype=np.float32)
    raw_slice = raw[:, :, args.slice_idx]

    norm = robust_normalize(raw)
    norm_slice = norm[:, :, args.slice_idx]
    head_mask = make_head_mask(norm_slice, thr=args.head_mask_thr, dilate_iters=args.head_mask_dilate)
    bbox = bbox_from_mask(head_mask, margin=args.fg_bbox_margin)

    cropped = crop_with_bbox(norm_slice, bbox)
    cropped_mask = crop_with_bbox(head_mask.astype(np.uint8), bbox).astype(bool)
    suppressed = cropped.copy()
    suppressed[~cropped_mask] = 0.0
    final = resize_to_hw(suppressed, tuple(args.out_hw))

    panels = [
        ("Raw T1 slice", image_from_array(raw_slice)),
        ("Robust normalized", image_from_array(norm_slice)),
        ("Head mask overlay", overlay_mask(norm_slice, head_mask)),
        ("Head bbox crop", draw_bbox(norm_slice, bbox)),
        ("Background suppressed", image_from_array(suppressed)),
        ("Final 256x256 input", image_from_array(final)),
    ]
    title = f"Preprocessing stages | {args.subject_id} | axial slice={args.slice_idx}"
    out_path = Path(args.out)
    if args.format == "pdf":
        out_path = out_path.with_suffix(".pdf")
        save_panel(panels, out_path.with_suffix(".png"), title, save_pdf=True)
        out_path.with_suffix(".png").unlink(missing_ok=True)
    else:
        save_panel(panels, out_path, title, save_pdf=args.format == "both")


if __name__ == "__main__":
    main()
