"""
Lightweight baseline analysis helpers for audio steganography demos.

Waveform/spectrogram plots, BER, runtime timing, and markdown-style summaries.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# BER
# ---------------------------------------------------------------------------

def text_to_bits(text: str) -> str:
    """UTF-8 bytes as a binary string (8 bits per byte, MSB first)."""
    return "".join(format(b, "08b") for b in text.encode("utf-8"))


def bit_error_rate(
    original: str,
    recovered: str,
    *,
    to_bits: Callable[[str], str] | None = None,
) -> float:
    """
    Bit error rate in percent between original and recovered payloads.

    Compares bit strings over min(length); extra bits in either string are ignored.
    """
    encode = to_bits or text_to_bits
    expected = encode(original)
    actual = encode(recovered)
    length = min(len(expected), len(actual))
    if length == 0:
        return 0.0
    errors = sum(1 for i in range(length) if expected[i] != actual[i])
    return 100.0 * errors / length


def print_ber(
    original: str,
    recovered: str,
    *,
    to_bits: Callable[[str], str] | None = None,
) -> float:
    """Compute BER and print a clear one-line summary."""
    pct = bit_error_rate(original, recovered, to_bits=to_bits)
    n_bits = min(len(text_to_bits(original)), len(text_to_bits(recovered)))
    n_err = int(round(pct * n_bits / 100.0)) if n_bits else 0
    print(f"BER: {pct:.4f}%  ({n_err} bit errors / {n_bits} compared bits)")
    return pct


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------

def time_embed_extract(
    embed_fn: Callable[[], Any],
    extract_fn: Callable[[], Any],
) -> tuple[float, float, Any, Any]:
    """Measure embedding and extraction separately (seconds, perf_counter)."""
    t0 = time.perf_counter()
    stego_or_side = embed_fn()
    embed_sec = time.perf_counter() - t0

    t1 = time.perf_counter()
    recovered = extract_fn()
    extract_sec = time.perf_counter() - t1

    return embed_sec, extract_sec, stego_or_side, recovered


def print_runtime(embed_sec: float, extract_sec: float) -> None:
    print(f"Embedding time:   {embed_sec * 1000:.2f} ms")
    print(f"Extraction time:  {extract_sec * 1000:.2f} ms")
    print(f"Total time:       {(embed_sec + extract_sec) * 1000:.2f} ms")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def mono(y: np.ndarray) -> np.ndarray:
    """First channel or 1D vector for plotting."""
    y = np.asarray(y, dtype=np.float64)
    return y[:, 0] if y.ndim == 2 else y


def plot_waveform_comparison(
    cover: np.ndarray,
    stego: np.ndarray,
    sr: int,
    *,
    segment_sec: float = 0.05,
    title: str = "Waveform (segment)",
    save_path: Path | None = None,
) -> None:
    """Side-by-side waveform plot of a short segment for visibility."""
    cover_m = mono(cover)
    stego_m = mono(stego)
    n = min(len(cover_m), len(stego_m))
    n_seg = min(n, int(segment_sec * sr))
    t = np.arange(n_seg) / sr

    fig, axes = plt.subplots(1, 2, figsize=(10, 3), sharey=True)
    axes[0].plot(t, cover_m[:n_seg], color="steelblue", linewidth=0.8)
    axes[0].set_title("Cover (original)")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")

    axes[1].plot(t, stego_m[:n_seg], color="darkorange", linewidth=0.8)
    axes[1].set_title("Stego")
    axes[1].set_xlabel("Time (s)")

    fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Saved waveform plot: {save_path}")
    plt.close(fig)


def plot_spectrogram_comparison(
    cover: np.ndarray,
    stego: np.ndarray,
    sr: int,
    *,
    title: str = "Spectrogram comparison",
    save_path: Path | None = None,
    n_fft: int = 1024,
    hop_length: int = 512,
) -> None:
    """Side-by-side log-magnitude spectrograms (librosa.display.specshow)."""
    cover_m = mono(cover)
    stego_m = mono(stego)
    n = min(len(cover_m), len(stego_m))
    cover_m = cover_m[:n]
    stego_m = stego_m[:n]

    s_cover = np.abs(librosa.stft(cover_m, n_fft=n_fft, hop_length=hop_length))
    s_stego = np.abs(librosa.stft(stego_m, n_fft=n_fft, hop_length=hop_length))
    s_cover_db = librosa.amplitude_to_db(s_cover, ref=np.max)
    s_stego_db = librosa.amplitude_to_db(s_stego, ref=np.max)
    vmin = min(s_cover_db.min(), s_stego_db.min())
    vmax = max(s_cover_db.max(), s_stego_db.max())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True, constrained_layout=True)
    img0 = librosa.display.specshow(
        s_cover_db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="hz",
        ax=axes[0],
        vmin=vmin,
        vmax=vmax,
    )
    axes[0].set_title("Cover (original)")
    img1 = librosa.display.specshow(
        s_stego_db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="hz",
        ax=axes[1],
        vmin=vmin,
        vmax=vmax,
    )
    axes[1].set_title("Stego")
    fig.colorbar(img1, ax=axes, format="%+2.0f dB", label="Magnitude (dB)")
    fig.suptitle(title)
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Saved spectrogram plot: {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Markdown-style report (printed to console)
# ---------------------------------------------------------------------------

METHOD_NOTES: dict[str, dict[str, str]] = {
    "LSB": {
        "observations": (
            "LSB changes only the least significant bit of PCM samples; "
            "waveforms often look identical in short segments. "
            "BER is 0% on lossless round-trip with the correct password."
        ),
        "limitations": (
            "Fragile to re-quantization, noise, and lossy compression; "
            "low perceptual masking; security depends on XOR scrambling, not robustness."
        ),
        "improvements": (
            "Spread LSB across frequency bands, use adaptive bit locations, "
            "or combine with encryption and error-correcting codes."
        ),
    },
    "Echo Hiding": {
        "observations": (
            "Echo hiding adds delayed copies per frame; spectrograms may show subtle "
            "time-frequency texture changes. Cepstrum-based decode works best on "
            "covers without strong periodic bias."
        ),
        "limitations": (
            "Low capacity (one bit per frame), fixed delays (d0, d1), "
            "and sensitivity to cover type and MP3."
        ),
        "improvements": (
            "Adaptive echo amplitudes, spread-spectrum kernels, "
            "and multi-band or keyed delay patterns."
        ),
    },
    "DWT": {
        "observations": (
            "Payload overwrites finest detail wavelet coefficients; "
            "audible distortion grows with payload energy. "
            "BER on text depends on matching scaled to subband level."
        ),
        "limitations": (
            "Custom detail-only cascade with truncated coefficient lengths; "
            "approximate re-analysis; fixed db12 band."
        ),
        "improvements": (
            "Adaptive subband selection, QIM embedding, "
            "and perceptual masking to set scaled per frame."
        ),
    },
    "Cepstrum": {
        "observations": (
            "Each bit nudges a fixed quefrency coefficient; "
            "stego waveforms can remain very close to cover in short views. "
            "BER rises if strength is too low or cover is too short."
        ),
        "limitations": (
            "Fixed quefrency index, one bit per frame, "
            "phase not modified (magnitude-only homomorphic path)."
        ),
        "improvements": (
            "Adaptive quefrency masks, spread-spectrum cepstral embedding, "
            "and overlap-add framing to reduce block edges."
        ),
    },
}


def print_analysis_report(method: str) -> None:
    """Print Observations / Limitations / Possible Improvements for a method."""
    notes = METHOD_NOTES.get(method, METHOD_NOTES["LSB"])
    print("\n--- Observations ---")
    print(notes["observations"])
    print("\n--- Limitations ---")
    print(notes["limitations"])
    print("\n--- Possible Improvements ---")
    print(notes["improvements"])
    print()


def run_baseline_visualization(
    cover: np.ndarray,
    stego: np.ndarray,
    sr: int,
    method: str,
    out_dir: Path,
) -> None:
    """Generate waveform + spectrogram comparison plots for one method."""
    tag = method.lower().replace(" ", "_")
    plot_waveform_comparison(
        cover,
        stego,
        sr,
        title=f"{method} — waveform segment",
        save_path=out_dir / f"{tag}_waveform.png",
    )
    plot_spectrogram_comparison(
        cover,
        stego,
        sr,
        title=f"{method} — spectrogram",
        save_path=out_dir / f"{tag}_spectrogram.png",
    )
