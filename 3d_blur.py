# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 14:27:19 2025

@author: davidcaldwell
"""

import os
import numpy as np
import pydicom
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

# Reference frame rate used for the relative temporal correction.
FRAME_RATE_REF = 15.0

# Manually specified acquisition frame rate (FPS) for all datasets.
# Change this value as needed.
FRAME_RATE_MANUAL = 7.5

# ---------------------------------------------------------------------------
# DICOM loading
# ---------------------------------------------------------------------------

def load_dicom_frames(dicom_path: str) -> tuple[np.ndarray, float]:
    """
    Load a multi-frame DICOM file and return grayscale frames with a
    manually specified acquisition frame rate.

    RGB pixel data (T, H, W, 3) is converted to grayscale using standard
    luminance weights. Single-channel data (T, H, W) is returned as-is.

    Returns
    -------
    frames : np.ndarray, shape (T, H, W), dtype uint8
    frame_rate : float, frames per second (from FRAME_RATE_MANUAL)
    """
    ds = pydicom.dcmread(dicom_path)

    if "PixelData" not in ds:
        raise ValueError(f"No pixel data in {dicom_path}")

    pixels = ds.pixel_array

    if pixels.ndim == 4 and pixels.shape[-1] == 3:
        frames = (
            0.299 * pixels[..., 0] +
            0.587 * pixels[..., 1] +
            0.114 * pixels[..., 2]
        ).astype(np.uint8)
    elif pixels.ndim == 3:
        frames = pixels.astype(np.uint8)
    else:
        raise ValueError(
            f"Unexpected pixel array shape {pixels.shape}. "
            "Expected (T, H, W) or (T, H, W, 3)."
        )

    # Use the manually specified frame rate for all datasets
    frame_rate = FRAME_RATE_MANUAL
    return frames, frame_rate


# ---------------------------------------------------------------------------
# BM3D calculation
# ---------------------------------------------------------------------------

def compute_bm3d(volume: np.ndarray, frame_rate: float) -> float:
    """
    Compute the 3D Blur Metric (BM3D) for a grayscale spatio-temporal volume.

    The volume is convolved with a 3x3x3 averaging filter to produce a blurred
    reference. Absolute first-order differences are then computed along all
    three axes (t, y, x) for both the original and blurred volumes. Edge
    strength in each direction is the positive part of (original diff - blurred
    diff): a large value means the edge was present in the original and removed
    by blurring, while near-zero means it was already blurred. BM3D is the
    fraction of total edge energy that has been suppressed by blurring.

    Temporal differences are scaled by (FRAME_RATE_REF / frame_rate) before
    summation. This upweights lower frame rate acquisitions to correct for the
    fact that they naturally produce larger inter-frame differences due to
    greater temporal spacing, not greater blur. Both the original and blurred
    temporal differences are scaled by the same factor so the edge strength
    subtraction remains consistent.

    Parameters
    ----------
    volume : np.ndarray, shape (T, H, W)
        Grayscale spatio-temporal volume.
    frame_rate : float
        Acquisition frame rate in FPS, read from the DICOM header.

    Returns
    -------
    float
        BM3D value in [0, 1]. Lower values indicate less blurring.
    """
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume (T, H, W), got shape {volume.shape}.")

    # Work in int32 to avoid uint8 overflow when computing pixel differences
    F = np.clip(volume.astype(np.int32), 0, 255)

    # Blurred reference: convolve with a normalised 3x3x3 averaging kernel
    h = np.ones((3, 3, 3), dtype=np.float32) / 27.0
    B = convolve(F, h)

    # Absolute first-order differences along t (temporal), y (height), x (width)
    dFt = np.abs(F[1:, :,  :] - F[:-1, :,  :]).astype(np.float32)
    dFy = np.abs(F[:,  1:, :] - F[:,  :-1, :]).astype(np.float32)
    dFx = np.abs(F[:,  :,  1:] - F[:,  :,  :-1]).astype(np.float32)

    dBt = np.abs(B[1:, :,  :] - B[:-1, :,  :]).astype(np.float32)
    dBy = np.abs(B[:,  1:, :] - B[:,  :-1, :]).astype(np.float32)
    dBx = np.abs(B[:,  :,  1:] - B[:,  :,  :-1]).astype(np.float32)

    # Relative temporal correction: scale by fr_ref / fr.
    # At 15 FPS (fr_ref) the factor is 1.0; at 7.5 FPS it is 2.0.
    # Both dFt and dBt are scaled by the same factor so their difference
    # (the edge strength Vt) remains dimensionally consistent.
    temporal_scale = FRAME_RATE_REF / frame_rate
    dFt_scaled = dFt * temporal_scale
    dBt_scaled = dBt * temporal_scale

    # Edge strength: only retain directions where the original was sharper than
    # the blurred version. Negative values (already blurred) are clamped to 0.
    Vt = np.maximum(dFt_scaled - dBt_scaled, 0.0)
    Vy = np.maximum(dFy - dBy, 0.0)
    Vx = np.maximum(dFx - dBx, 0.0)

    sum_F = np.sum(dFt_scaled) + np.sum(dFy) + np.sum(dFx)
    sum_V = np.sum(Vt) + np.sum(Vy) + np.sum(Vx)

    if sum_F == 0.0:
        # Completely uniform volume — no edges to assess
        return 0.0

    return round(float((sum_F - sum_V) / sum_F), 4)


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def process_dicom_files(dicom_paths: list[str]) -> list[dict]:
    """
    Compute BM3D for a list of DICOM files.

    Returns a list of result dicts, each with keys:
        'name'       : filename
        'frame_rate' : FPS read from DICOM header
        'bm3d'       : BM3D value, or None if the file could not be processed
        'error'      : error message string, or None
    """
    results = []

    for path in dicom_paths:
        name = os.path.basename(path)

        if not os.path.exists(path):
            print(f"File not found: {path}")
            results.append({"name": name, "frame_rate": None, "bm3d": None, "error": "File not found"})
            continue

        try:
            frames, frame_rate = load_dicom_frames(path)
            print(f"{name}: {frames.shape[0]} frames at {frame_rate} FPS")

            bm3d_value = compute_bm3d(frames, frame_rate)
            print(f"  BM3D = {bm3d_value}")

            results.append({"name": name, "frame_rate": frame_rate, "bm3d": bm3d_value, "error": None})

        except Exception as e:
            print(f"  Error processing {name}: {e}")
            results.append({"name": name, "frame_rate": None, "bm3d": None, "error": str(e)})

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_bm3d_results(results: list[dict]) -> None:
    """
    Bar chart of BM3D values. Files that could not be processed are omitted.
    """
    valid = [r for r in results if r["bm3d"] is not None]
    if not valid:
        print("No valid results to plot.")
        return

    values = [r["bm3d"] for r in valid]
    labels = [f"{r['name']}\n({r['frame_rate']} FPS)" for r in valid]

    fig, ax = plt.subplots(figsize=(max(8, len(valid) * 1.5), 5))
    bars = ax.bar(labels, values, color="steelblue", edgecolor="white", width=0.6)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=9
        )

    ax.set_xlabel("Dataset")
    ax.set_ylabel("BM3D")
    ax.set_title("3D Blur Metric by dataset")
    ax.set_ylim(0, max(values) * 1.15)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# Program is expecting DICOM fluoroscopy sequences (e.g., algorithm_1.dcm, algorithm_2.dcm, etc.)

if __name__ == "__main__":
    dicom_paths = [
        "algorithm_1.dcm",
        "algorithm_2.dcm",
        "algorithm_3.dcm",
        "algorithm_4.dcm",
    ]

    results = process_dicom_files(dicom_paths)
    plot_bm3d_results(results)
