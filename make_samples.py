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

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ---------------------------------------------------------------------------
# Image utility: letterbox (BGR-native core)
# ---------------------------------------------------------------------------


def letterbox_resize_bgr(
    bgr_img: np.ndarray, target_w: int = 640, target_h: int = 640
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Letterbox-resize an HxWx3 **BGR** uint8 image to (target_h, target_w) with gray padding.

    Returns:
        bgr_padded:  (target_h, target_w, 3) uint8, BGR order (OpenCV-native)
        rgb_padded:  (target_h, target_w, 3) uint8, RGB order
        nv12_padded: (target_h*3//2, target_w) uint8, NV12 (Y plane + interleaved UV plane)

    Notes:
        - Resizing/padding is performed on BGR (OpenCV-native).
        - RGB is derived once at the end (for raw .rgb output).
        - NV12 is generated via fallback path: BGR -> I420 -> NV12 repack.
    """
    if bgr_img.ndim != 3 or bgr_img.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 BGR; got shape {bgr_img.shape}")
    if bgr_img.dtype != np.uint8:
        raise TypeError(f"Expected uint8 BGR image; got dtype {bgr_img.dtype}")

    bgr = np.ascontiguousarray(bgr_img)

    # Compute letterbox on BGR
    h0, w0 = bgr.shape[:2]
    r = min(target_w / float(w0), target_h / float(h0))
    new_w = int(round(w0 * r))
    new_h = int(round(h0 * r))

    resized = (
        bgr
        if (w0, h0) == (new_w, new_h)
        else cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    )

    dw = target_w - new_w
    dh = target_h - new_h
    pad_left = dw // 2
    pad_right = dw - pad_left
    pad_top = dh // 2
    pad_bottom = dh - pad_top

    bgr_padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    bgr_padded = np.ascontiguousarray(bgr_padded)

    # -----------------------
    # Validation: padded size
    # -----------------------
    if bgr_padded.shape[:2] != (target_h, target_w):
        raise RuntimeError(
            f"Letterbox produced unexpected size: got {bgr_padded.shape[:2]}, "
            f"expected {(target_h, target_w)}"
        )
    if bgr_padded.dtype != np.uint8 or bgr_padded.ndim != 3 or bgr_padded.shape[2] != 3:
        raise RuntimeError(
            f"Unexpected padded BGR format: shape={bgr_padded.shape}, dtype={bgr_padded.dtype}"
        )

    # Derive RGB once (contiguous)
    rgb_padded = np.ascontiguousarray(bgr_padded[..., ::-1])

    # Fallback NV12 generation: BGR -> I420 -> NV12 repack (OpenCV layout: (H*3//2, W))
    i420 = cv2.cvtColor(bgr_padded, cv2.COLOR_BGR2YUV_I420)
    H, W = target_h, target_w

    # -----------------------
    # Validation: even dims
    # -----------------------
    if (H % 2) != 0 or (W % 2) != 0:
        raise RuntimeError(f"I420/NV12 require even H/W. Got H={H}, W={W}")

    # --------------------------------------
    # Validation: I420 buffer size and shape
    # --------------------------------------
    expected_size = H * W * 3 // 2
    if i420.dtype != np.uint8:
        raise RuntimeError(f"Unexpected I420 dtype: {i420.dtype} (expected uint8)")
    if i420.size != expected_size:
        raise RuntimeError(
            f"Unexpected I420 size: got {i420.size}, expected {expected_size} "
            f"(i420.shape={i420.shape}, H={H}, W={W})"
        )
    expected_i420_shape = (H * 3 // 2, W)
    if i420.ndim != 2 or i420.shape != expected_i420_shape:
        raise RuntimeError(
            f"Unexpected I420 shape: got {i420.shape}, expected {expected_i420_shape}"
        )

    # Y plane (H, W)
    y_plane = i420[:H, :]

    # U and V planes (each is H/2 * W/2 bytes), stored consecutively after Y
    u_rows = H // 4
    v_rows = H // 4
    u_slice = i420[H : H + u_rows, :]
    v_slice = i420[H + u_rows : H + u_rows + v_rows, :]

    # -----------------------
    # Validation: U/V slices
    # -----------------------
    if u_slice.size != (H // 2) * (W // 2):
        raise RuntimeError(
            f"Unexpected U slice size: got {u_slice.size}, expected {(H // 2) * (W // 2)} "
            f"(u_slice.shape={u_slice.shape})"
        )
    if v_slice.size != (H // 2) * (W // 2):
        raise RuntimeError(
            f"Unexpected V slice size: got {v_slice.size}, expected {(H // 2) * (W // 2)} "
            f"(v_slice.shape={v_slice.shape})"
        )

    u = u_slice.reshape(H // 2, W // 2)
    v = v_slice.reshape(H // 2, W // 2)

    nv12_padded = np.empty((H * 3 // 2, W), dtype=np.uint8)
    nv12_padded[:H, :] = y_plane

    # Interleaved UV plane (H/2, W): U in even columns, V in odd columns
    uv = nv12_padded[H:, :].reshape(H // 2, W)
    uv[:, 0::2] = u
    uv[:, 1::2] = v

    # -----------------------
    # Validation: NV12 output
    # -----------------------
    if nv12_padded.shape != (H * 3 // 2, W):
        raise RuntimeError(
            f"Unexpected NV12 shape: got {nv12_padded.shape}, expected {(H * 3 // 2, W)}"
        )

    return bgr_padded, rgb_padded, np.ascontiguousarray(nv12_padded)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def list_images(indir: Path) -> List[Path]:
    """List image files (non-recursive) in 'indir' matching known extensions."""
    if not indir.exists() or not indir.is_dir():
        raise NotADirectoryError(
            f"Input directory not found or not a directory: {indir}"
        )
    imgs = [
        p for p in indir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
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
    target_width: int,
    target_height: int,
    num_images: int,
) -> Tuple[int, int]:
    """
    Read images from 'indir', letterbox-resize and save outputs into a freshly
    created 'outdir' using the given filename 'prefix' and no zero padding.

    For each processed image, this function writes:
      - Raw RGB:   <prefix><i+1>.rgb      (letterboxed RGB bytes via numpy.tofile)
      - Raw NV12:  <prefix><i+1>.nv12     (letterboxed NV12 bytes via numpy.tofile)

    Notes:
      - The output directory is deleted and recreated on each run.
      - 'saved' counts the number of raw files successfully written (RGB and NV12), not images.
    """
    paths = list_images(indir)
    total_available = len(paths)
    to_save = max(0, min(int(num_images), total_available))

    logging.info(
        f"Found {total_available} image(s) in {indir}. Requested: {num_images}. Will save: {to_save}"
    )
    for idx, p in enumerate(paths[:to_save]):
        logging.debug(f"[{idx + 1}/{to_save}] {p.name}")

    # Always recreate output directory
    reset_outdir(outdir)

    saved = 0
    for i, p in enumerate(paths[:to_save]):
        # Read as BGR (OpenCV default)
        bgr_in = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr_in is None:
            logging.error(f"Failed to read image: {p}")
            continue

        logging.info(f"Opened BGR shape for {p.name}: {bgr_in.shape}")

        if bgr_in.dtype != np.uint8 or bgr_in.ndim != 3 or bgr_in.shape[-1] != 3:
            logging.error(
                f"Unexpected image format (expect HxWx3 uint8): {p} "
                f"shape={None if bgr_in is None else bgr_in.shape}"
            )
            continue

        # Avoidable channel swaps fixed: call BGR-native letterbox directly
        bgr_out, rgb_out, nv12_out = letterbox_resize_bgr(
            bgr_in, target_width, target_height
        )

        logging.info(
            f"letterbox outputs for {p.name}: "
            f"bgr_out={bgr_out.shape}, rgb_out={rgb_out.shape}, nv12_out={nv12_out.shape}"
        )

        # Write raw RGB and raw NV12 buffers
        rgb_path = outdir / f"{prefix}{i + 1}.rgb"
        nv12_path = outdir / f"{prefix}{i + 1}.nv12"
        try:
            rgb_out.tofile(str(rgb_path))
        except Exception as e:
            logging.error(f"Failed to write raw RGB file '{rgb_path}': {e}")
        else:
            saved += 1
        try:
            nv12_out.tofile(str(nv12_path))
        except Exception as e:
            logging.error(f"Failed to write raw NV12 file '{nv12_path}': {e}")
        else:
            saved += 1

    return saved, to_save


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    args = argparse.ArgumentParser(
        description="Read images from a folder, letterbox-resize to and save as raw RGB + raw NV12."
    )
    args.add_argument(
        "-i",
        "--indir",
        default="./test_images",
        help="Input folder with images (default: ./test_images)",
    )
    args.add_argument(
        "-o",
        "--outdir",
        default="./build/samples_640",
        help="Output directory (default: ./build/samples_640)",
    )
    args.add_argument(
        "--prefix", default="img", help="Filename prefix for outputs (default: img)"
    )
    args.add_argument(
        "-tw",
        "--target_width",
        type=int,
        default=640,
        help="Target width (default: 640)",
    )
    args.add_argument(
        "-th",
        "--target_height",
        type=int,
        default=640,
        help="Target height (default: 640)",
    )
    args.add_argument(
        "-m",
        "--num_images",
        type=int,
        default=10,
        help="Max number of images to process (default: 10)",
    )
    return args


def main() -> None:
    args = build_argparser().parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    indir = Path(args.indir)
    outdir = Path(args.outdir)

    logging.info(f"Input dir:  {indir.resolve()}")
    logging.info(f"Output dir: {outdir.resolve()}")

    saved, attempted = save_letterboxed_from_folder(
        indir=indir,
        outdir=outdir,
        prefix=args.prefix,
        target_width=args.target_width,
        target_height=args.target_height,
        num_images=args.num_images,
    )

    logging.info(
        f"Done. Wrote {saved} raw file(s) (RGB and/or NV12) for {attempted} image(s) from '{indir.name}'."
    )


if __name__ == "__main__":
    main()
