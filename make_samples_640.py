#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
**************************************************************************
||                        SiMa.ai CONFIDENTIAL                          ||
||   Unpublished Copyright (c) 2022-2023 SiMa.ai, All Rights Reserved.  ||
**************************************************************************
 NOTICE:  All information contained herein is, and remains the property of
 SiMa.ai. The intellectual and technical concepts contained herein are 
 proprietary to SiMa and may be covered by U.S. and Foreign Patents, 
 patents in process, and are protected by trade secret or copyright law.

 Dissemination of this information or reproduction of this material is 
 strictly forbidden unless prior written permission is obtained from 
 SiMa.ai.  Access to the source code contained herein is hereby forbidden
 to anyone except current SiMa.ai employees, managers or contractors who 
 have executed Confidentiality and Non-disclosure agreements explicitly 
 covering such access.

 The copyright notice above does not evidence any actual or intended 
 publication or disclosure  of  this
"""

import argparse
import logging
import shutil
from pathlib import Path
from typing import List, Tuple

import numpy as np
import cv2


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INPUT_W: int = 640
INPUT_H: int = 640
JPEG_QUALITY: int = 95
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ---------------------------------------------------------------------------
# Image utility: letterbox (expects RGB uint8; returns BGR + RGB)
# ---------------------------------------------------------------------------

def letterbox_resize_color(rgb_img: np.ndarray, target_w: int = INPUT_W, target_h: int = INPUT_H) -> Tuple[np.ndarray, np.ndarray]:
    """
    Letterbox-resize an HxWx3 **RGB** uint8 image to (target_h, target_w) with black padding.

    Input:
        rgb_img: HxWx3, dtype=uint8, RGB order.

    Returns:
        bgr_padded: HxWx3, uint8, BGR order (for OpenCV).
        rgb_padded: HxWx3, uint8, RGB order.

    Notes:
        - Resizing/padding is performed on the BGR image (OpenCV-native), then we derive RGB.
        - Both outputs are contiguous arrays.
    """
    if rgb_img.ndim != 3 or rgb_img.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 RGB; got shape {rgb_img.shape}")
    if rgb_img.dtype != np.uint8:
        raise TypeError(f"Expected uint8 RGB image; got dtype {rgb_img.dtype}")

    # RGB -> BGR for OpenCV ops
    bgr = rgb_img[..., ::-1]

    # Compute letterbox on BGR
    h0, w0 = bgr.shape[:2]
    r = min(target_w / float(w0), target_h / float(h0))
    new_w = int(round(w0 * r))
    new_h = int(round(h0 * r))

    resized = bgr if (w0, h0) == (new_w, new_h) else cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    dw = target_w - new_w
    dh = target_h - new_h
    pad_left = dw // 2
    pad_right = dw - pad_left
    pad_top = dh // 2
    pad_bottom = dh - pad_top

    bgr_padded = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

    # Also return RGB
    rgb_padded = bgr_padded[..., ::-1]

    return np.ascontiguousarray(bgr_padded), np.ascontiguousarray(rgb_padded)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def list_images(indir: Path) -> List[Path]:
    """List image files (non-recursive) in 'indir' matching known extensions."""
    if not indir.exists() or not indir.is_dir():
        raise NotADirectoryError(f"Input directory not found or not a directory: {indir}")
    imgs = [p for p in indir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    imgs.sort()
    return imgs


def reset_outdir(outdir: Path) -> None:
    """Recreate the output directory (delete if exists, then mkdir)."""
    if outdir.exists():
        logging.info(f"Removing existing output directory: {outdir.resolve()}")
        shutil.rmtree(outdir, ignore_errors=False)
    outdir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Created output directory: {outdir.resolve()}")


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def save_letterboxed_from_folder(
    indir: Path,
    outdir: Path,
    prefix: str,
    num_images: int,
    jpeg_quality: int = JPEG_QUALITY,
) -> Tuple[int, int]:
    """
    Read images from 'indir', assume RGB content semantics, letterbox to 640x640,
    and save JPEGs to a freshly created 'outdir' with 'prefix' and no zero padding.
    Also writes the RGB raw buffer as <prefix><i>.rgb via numpy.tofile().
    """
    paths = list_images(indir)
    total_available = len(paths)
    to_save = max(0, min(int(num_images), total_available))

    logging.info(f"Found {total_available} image(s) in {indir}. Requested: {num_images}. Will save: {to_save}")
    for idx, p in enumerate(paths[:to_save]):
        logging.debug(f"[{idx+1}/{to_save}] {p.name}")

    # Always recreate output directory
    reset_outdir(outdir)

    saved = 0
    for i, p in enumerate(paths[:to_save]):
        # Read as BGR (OpenCV default) then convert to RGB for our API
        bgr_in = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr_in is None:
            logging.error(f"Failed to read image: {p}")
            continue

        if bgr_in.dtype != np.uint8 or bgr_in.ndim != 3 or bgr_in.shape[-1] != 3:
            logging.error(f"Unexpected image format (expect HxWx3 uint8): {p} shape={None if bgr_in is None else bgr_in.shape}")
            continue

        rgb = bgr_in[..., ::-1]  # BGR -> RGB
        bgr_out, rgb_out = letterbox_resize_color(rgb, INPUT_W, INPUT_H)  # returns (BGR, RGB)

        # Write JPEG (BGR)
        jpg_path = outdir / f"{prefix}{i}.jpg"  # no leading zeros
        ok = cv2.imwrite(str(jpg_path), bgr_out, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        if not ok:
            logging.error(f"Failed to write: {jpg_path}")
            continue

        # Write raw RGB bytes
        rgb_path = outdir / f"{prefix}{i}.rgb"
        try:
            # rgb_out is contiguous uint8 HxWx3
            rgb_out.tofile(str(rgb_path))
        except Exception as e:
            logging.error(f"Failed to write raw RGB file '{rgb_path}': {e}")
            # still count the JPEG as saved
        else:
            saved += 1

    return saved, to_save


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Read RGB uint8 images from a folder, letterbox-resize to 640x640, and save as JPEG + raw RGB."
    )
    p.add_argument("-i", "--indir", default="./test_images", help="Input folder with images (default: ./test_images)")
    p.add_argument("-o", "--outdir", default="./build/samples_640", help="Output directory (default: ./build/samples_640)")
    p.add_argument("--prefix", default="img", help="Filename prefix for outputs (default: img)")
    p.add_argument("-m", "--num_images", type=int, default=5, help="Max number of images to process (default: 5)")
    p.add_argument("--quality", type=int, default=JPEG_QUALITY, help=f"JPEG quality (default: {JPEG_QUALITY})")
    p.add_argument("--log", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Log level (default: INFO)")
    return p


def main() -> None:
    args = build_argparser().parse_args()

    logging.basicConfig(level=getattr(logging, args.log), format="%(levelname)s: %(message)s")

    indir = Path(args.indir)
    outdir = Path(args.outdir)

    logging.info(f"Input dir:  {indir.resolve()}")
    logging.info(f"Output dir: {outdir.resolve()}")

    saved, attempted = save_letterboxed_from_folder(
        indir=indir,
        outdir=outdir,
        prefix=args.prefix,
        num_images=args.num_images,
        jpeg_quality=args.quality,
    )

    logging.info(f"Done. Saved {saved}/{attempted} pairs (JPEG + .rgb) from '{indir.name}'.")


if __name__ == "__main__":
    main()
