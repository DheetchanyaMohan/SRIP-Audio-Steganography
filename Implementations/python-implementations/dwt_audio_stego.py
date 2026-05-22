"""
DWT / wavelet audio steganography — Python port of Image-in-Audio-Steganography
(functions/encryptionDWT.m, decryptionDWT.m).

Embeds a 1D payload in the finest detail subband of a custom detail-only DWT cascade
(db12). The full project also maps images via ISTFT + 2D permutation (encryption2D.m);
this module implements the wavelet embed/extract core and optional text/byte helpers.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pywt
import soundfile as sf

WAVELET = "db12"
DWT_MODE = "sym"  # match MATLAB Wavelet Toolbox default extension for dwt/idwt


# ---------------------------------------------------------------------------
# Detail-only DWT cascade (MATLAB encryptionDWT / decryptionDWT loops)
# ---------------------------------------------------------------------------

def _decomposition_level(n_samples: int, n_embedded: int) -> int:
    """Number of cascade levels: floor(log2(n_samples / n_embedded))."""
    return int(np.floor(np.log2(n_samples / n_embedded)))


def decompose_detail_cascade(
    audio: np.ndarray,
    n_samples: int,
    n_embedded: int,
) -> tuple[np.ndarray, int]:
    """
    Repeatedly apply one-level DWT to the current *detail* branch (not standard Mallat).

    MATLAB stores coefficients in rows 1..2*level+1; row 2*level+1 is the finest
    detail subband where the payload is written.

    Returns
    -------
    coef : ndarray, shape (2*level+1, n_samples)
    level : int
    """
    level = _decomposition_level(n_samples, n_embedded)
    rows = 2 * level + 1
    coef = np.zeros((rows, n_samples), dtype=np.float64)

    # Row 0 == MATLAB coef(1,:) — approximation path starts as full audio
    coef[0, :] = audio[:n_samples]
    row_idx = 0
    cur_size = n_samples

    for _ in range(level):
        segment = coef[row_idx, :cur_size]
        # One DWT split: cA (low) and cD (high) subband coefficients
        c_a, c_d = pywt.dwt(segment, WAVELET, mode=DWT_MODE)
        coef[row_idx + 1, : len(c_a)] = c_a
        coef[row_idx + 2, : len(c_d)] = c_d
        row_idx += 2
        cur_size = int(np.ceil(cur_size / 2))

    return coef, level


def reconstruct_from_cascade(
    coef: np.ndarray,
    level: int,
    n_samples: int,
) -> np.ndarray:
    """
    Inverse of decompose_detail_cascade via successive idwt (MATLAB composition loop).

    Pairs rows (idx-1, idx) as (cA, cD), using cur_size = ceil(n_samples / 2^level)
    at the finest level, then doubling each reconstruction step.
    """
    row_idx = 2 * level  # finest detail row (MATLAB 2*level+1)
    cur_size = int(np.ceil(n_samples / (2**level)))

    for _ in range(level):
        c_a = coef[row_idx - 1, :cur_size]
        c_d = coef[row_idx, :cur_size]
        reconstructed = pywt.idwt(c_a, c_d, WAVELET, mode=DWT_MODE)
        coef[row_idx - 2, : len(reconstructed)] = reconstructed
        row_idx -= 2
        cur_size *= 2

    return coef[0, :n_samples].copy()


def finest_detail_row(level: int) -> int:
    """0-based row index of the embedding subband (MATLAB row 2*level+1)."""
    return 2 * level


# ---------------------------------------------------------------------------
# Payload helpers (text / bytes — optional; core MATLAB embeds ISTFT signal x)
# ---------------------------------------------------------------------------

def text_to_payload(text: str, n_embedded: int) -> np.ndarray:
    """Map UTF-8 bytes to [0, 1] floats; zero-pad to n_embedded."""
    raw = text.encode("utf-8")
    if len(raw) > n_embedded:
        warnings.warn("Text truncated to fit n_embedded.")
        raw = raw[:n_embedded]
    payload = np.zeros(n_embedded, dtype=np.float64)
    payload[: len(raw)] = np.frombuffer(raw, dtype=np.uint8) / 255.0
    return payload


def payload_to_text(payload: np.ndarray) -> str:
    """Recover UTF-8 string from normalized payload coefficients."""
    bytes_arr = np.clip(np.round(payload * 255.0), 0, 255).astype(np.uint8)
    # Trim trailing padding zeros
    end = len(bytes_arr)
    while end > 0 and bytes_arr[end - 1] == 0:
        end -= 1
    return bytes(bytes_arr[:end]).decode("utf-8", errors="replace")


def estimate_scale(
    cover: np.ndarray,
    n_samples: int,
    n_embedded: int,
    fraction: float = 0.9,
) -> float:
    """
    Suggest `scaled` so payload values sit near the native coefficient magnitude
    (improves extract fidelity for this truncated cascade).
    """
    coef, level = decompose_detail_cascade(cover, n_samples, n_embedded)
    band = coef[finest_detail_row(level), :n_embedded]
    peak = float(np.max(np.abs(band))) or 1.0
    return peak * fraction


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed_message(
    cover: np.ndarray,
    payload: np.ndarray,
    n_samples: int,
    n_embedded: int,
    scaled: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Embed a 1D payload into the cover audio (encryptionDWT.m).

    Parameters
    ----------
    cover : ndarray
        Cover waveform (1D or samples x channels; first channel used if 2D).
    payload : ndarray
        Secret signal x (length <= n_embedded), e.g. ISTFT of image or text_to_payload().
    n_samples : int
        Number of cover samples to process (MATLAB n_samples).
    n_embedded : int
        Length of finest detail band / max payload length (MATLAB n_embedded).
    scaled : float
        Payload is stored as payload/scaled in the wavelet domain; divide on embed,
        multiply on extract (MATLAB `scaled`).

    Returns
    -------
    stego : ndarray
        Time-domain stego signal (length n_samples on channel 0).
    steg_old : ndarray
        Original finest-detail coefficients before overwrite.
    steg_new : ndarray
        New finest-detail coefficients after overwrite.
    """
    if cover.ndim == 2:
        audio = cover[:, 0].astype(np.float64)
    else:
        audio = np.asarray(cover, dtype=np.float64)

    payload = np.asarray(payload, dtype=np.float64).ravel()
    if len(payload) > n_embedded:
        raise ValueError("payload length exceeds n_embedded")

    coef, level = decompose_detail_cascade(audio, n_samples, n_embedded)
    row = finest_detail_row(level)

    steg_old = coef[row, : len(payload)].copy()
    coef[row, : len(payload)] = payload / scaled
    steg_new = coef[row, : len(payload)].copy()

    stego = reconstruct_from_cascade(coef, level, n_samples)

    if cover.ndim == 2:
        out = cover.copy().astype(np.float64)
        out[:n_samples, 0] = stego
        return out, steg_old, steg_new

    return stego, steg_old, steg_new


def extract_message(
    stego: np.ndarray,
    n_samples: int,
    n_embedded: int,
    scaled: float = 1.0,
    payload_len: int | None = None,
) -> np.ndarray:
    """
    Extract payload from stego audio (decryptionDWT.m).

    Re-analyzes the stego with the same cascade and reads the finest detail row.
    Recovery is approximate if payload statistics differ strongly from the cover band.
    """
    if stego.ndim == 2:
        audio = stego[:, 0].astype(np.float64)
    else:
        audio = np.asarray(stego, dtype=np.float64)

    coef, level = decompose_detail_cascade(audio, n_samples, n_embedded)
    row = finest_detail_row(level)
    length = n_embedded if payload_len is None else min(payload_len, n_embedded)
    return coef[row, :length] * scaled


def embed_text(
    cover: np.ndarray,
    text: str,
    n_samples: int,
    n_embedded: int,
    scaled: float | None = None,
) -> tuple[np.ndarray, str]:
    """Convenience: embed UTF-8 text as normalized payload coefficients."""
    payload = text_to_payload(text, n_embedded)
    if scaled is None:
        audio = cover[:, 0] if cover.ndim == 2 else cover
        scaled = estimate_scale(audio, n_samples, n_embedded)
    stego, _, _ = embed_message(cover, payload, n_samples, n_embedded, scaled)
    return stego, text


def extract_text(
    stego: np.ndarray,
    n_samples: int,
    n_embedded: int,
    scaled: float,
    payload_len: int | None = None,
) -> str:
    """Convenience: extract and decode UTF-8 text payload."""
    payload = extract_message(
        stego, n_samples, n_embedded, scaled, payload_len=payload_len
    )
    if payload_len is not None:
        payload = payload[:payload_len]
    return payload_to_text(payload)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    data, fs = sf.read(path, dtype="float64", always_2d=True)
    return data, fs


def save_audio(path: str | Path, data: np.ndarray, fs: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, data, fs)


# ---------------------------------------------------------------------------
# Example usage + baseline experiment
# ---------------------------------------------------------------------------

def run_baseline_experiment(
    cover: np.ndarray,
    fs: int,
    message: str,
    n_samples: int = 65536,
    n_embedded: int = 8192,
    plot_dir: Path | None = None,
) -> tuple[np.ndarray, str]:
    """Embed/extract with timing, BER, plots, and printed analysis notes."""
    from stego_analysis import (
        print_analysis_report,
        print_ber,
        print_runtime,
        run_baseline_visualization,
        time_embed_extract,
    )

    plot_dir = plot_dir or Path(__file__).resolve().parent / "audio_out" / "analysis" / "dwt"
    payload = text_to_payload(message, n_embedded)
    scaled = estimate_scale(cover[:, 0] if cover.ndim == 2 else cover, n_samples, n_embedded)
    state: dict = {}

    def _embed():
        stego, _, _ = embed_message(cover.copy(), payload, n_samples, n_embedded, scaled)
        state["stego"] = stego
        return stego

    def _extract():
        raw = extract_message(
            state["stego"], n_samples, n_embedded, scaled, payload_len=n_embedded
        )
        n_bytes = len(message.encode("utf-8"))
        return payload_to_text(raw[:n_bytes])

    embed_sec, extract_sec, _, recovered = time_embed_extract(_embed, _extract)
    stego = state["stego"]

    print("\n=== DWT baseline experiment ===")
    print("Original :", message)
    print("Recovered:", recovered)
    print(f"scaled   : {scaled:.6f}")
    print_ber(message, recovered)
    print_runtime(embed_sec, extract_sec)

    run_baseline_visualization(cover, stego, fs, "DWT", plot_dir)
    print_analysis_report("DWT")
    return stego, recovered


if __name__ == "__main__":
    fs = 44100
    n_samples = 65536
    n_embedded = 8192

    rng = np.random.default_rng(0)
    t = np.arange(n_samples) / fs
    cover = (0.2 * np.sin(2 * np.pi * 440 * t) + 0.05 * rng.standard_normal(n_samples))[
        :, np.newaxis
    ]
    message = "Hidden in wavelet coefficients"

    stego, _ = run_baseline_experiment(cover, fs, message, n_samples, n_embedded)
    out = Path(__file__).resolve().parent / "audio_out" / "dwt_stego.wav"
    save_audio(out, stego, fs)
    print(f"Wrote {out}")
