"""
DWT / wavelet audio steganography — Python port of Image-in-Audio-Steganography
(functions/encryptionDWT.m, decryptionDWT.m).

Embeds a 1D payload in the finest detail subband of a custom detail-only DWT cascade
(db12). The full project also maps images via ISTFT + 2D permutation (encryption2D.m);
this module implements the wavelet embed/extract core and optional text/byte helpers.

Byte/text payloads are embedded as normalized coefficients (payload/scaled). Extraction
re-runs the same cascade on the stego signal, so recovery is approximate (analysis /
synthesis mismatch), not bit-exact like LSB. Float audio payloads matched to the
subband are the intended loss-tolerant use case from the original MATLAB flow.
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

def bytes_to_payload(data: bytes, n_embedded: int) -> np.ndarray:
    """Map raw bytes to [0, 1] floats; zero-pad to n_embedded."""
    raw = data
    if len(raw) > n_embedded:
        warnings.warn("Payload truncated to fit n_embedded.")
        raw = raw[:n_embedded]
    payload = np.zeros(n_embedded, dtype=np.float64)
    payload[: len(raw)] = np.frombuffer(raw, dtype=np.uint8) / 255.0
    return payload


def coeffs_to_bytes(payload: np.ndarray, *, nbytes: int | None = None) -> bytes:
    """Recover bytes from normalized wavelet payload coefficients (lossy decode)."""
    bytes_arr = np.clip(np.round(payload * 255.0), 0, 255).astype(np.uint8)
    if nbytes is not None:
        return bytes(bytes_arr[:nbytes])
    end = len(bytes_arr)
    while end > 0 and bytes_arr[end - 1] == 0:
        end -= 1
    return bytes(bytes_arr[:end])


def payload_to_bytes(payload: np.ndarray, *, nbytes: int | None = None) -> bytes:
    """Alias for coeffs_to_bytes (wavelet coefficients, not PayloadInfo)."""
    return coeffs_to_bytes(payload, nbytes=nbytes)


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

    Recovery is approximate: overwriting the finest detail band and reconstructing
    the waveform does not preserve coefficients under a second analysis pass.
    Use ``payload_len`` to limit how many coefficients are decoded to bytes.
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

def _payload_from_info(
    info: object,
    payload_bytes: bytes,
    n_embedded: int,
) -> tuple[np.ndarray, bytes]:
    """Build float wavelet payload and reference bytes for BER."""
    from stego_analysis import PayloadInfo

    if isinstance(info, PayloadInfo) and info.kind == "audio" and info.audio is not None:
        p = info.audio.astype(np.float64)
        if len(p) > n_embedded:
            p = p[:n_embedded]
        else:
            buf = np.zeros(n_embedded, dtype=np.float64)
            buf[: len(p)] = p
            p = buf
        peak = float(np.max(np.abs(p))) or 1.0
        p = p / peak
        ref = payload_bytes[: len(p)] if len(payload_bytes) >= len(p) else payload_bytes
        return p, ref

    return bytes_to_payload(payload_bytes, n_embedded), payload_bytes


def run_baseline_experiment(
    cover: np.ndarray,
    fs: int,
    payload_bytes: bytes,
    n_samples: int = 65536,
    n_embedded: int = 8192,
    plot_dir: Path | None = None,
    *,
    cover_path: Path | None = None,
    stego_path: Path | None = None,
    payload_info: object | None = None,
    extracted_path: Path | None = None,
) -> tuple[np.ndarray, bytes]:
    """Embed/extract with full evaluation framework."""
    from stego_analysis import (
        PayloadInfo,
        print_analysis_report,
        resolve_payload,
        run_full_baseline_evaluation,
        suggested_extracted_path,
        time_embed_extract,
    )

    plot_dir = plot_dir or Path(__file__).resolve().parent / "audio_out" / "analysis" / "dwt"
    pinfo = (
        payload_info
        if isinstance(payload_info, PayloadInfo)
        else resolve_payload(payload_text=payload_bytes.decode("utf-8", errors="replace"))
    )
    payload, ber_ref = _payload_from_info(pinfo, payload_bytes, n_embedded)
    n_keep = min(len(ber_ref), len(payload))
    scaled = estimate_scale(cover[:, 0] if cover.ndim == 2 else cover, n_samples, n_embedded)
    state: dict = {}

    def _embed():
        stego, _, _ = embed_message(cover.copy(), payload, n_samples, n_embedded, scaled)
        state["stego"] = stego
        return stego

    def _extract():
        raw = extract_message(
            state["stego"],
            n_samples,
            n_embedded,
            scaled,
            payload_len=n_keep,
        )
        return coeffs_to_bytes(raw, nbytes=n_keep)

    embed_sec, extract_sec, _, recovered = time_embed_extract(_embed, _extract)
    stego = state["stego"]

    if stego_path is not None:
        save_audio(stego_path, stego, fs)

    out_extract = extracted_path or suggested_extracted_path(
        "DWT", pinfo, Path(__file__).resolve().parent / "audio_out" / "extracted"
    )

    print("\n=== DWT baseline experiment ===")
    print(f"scaled: {scaled:.6f}")
    run_full_baseline_evaluation(
        "DWT",
        cover,
        stego,
        fs,
        ber_ref[:n_keep],
        recovered,
        embed_sec,
        extract_sec,
        plot_dir,
        cover_path=cover_path,
        stego_path=stego_path,
        payload_info=pinfo,
        extracted_path=out_extract,
    )
    print_analysis_report("DWT")
    return stego, recovered


if __name__ == "__main__":
    import argparse

    from stego_analysis import (
        add_experiment_arguments,
        load_cover_audio,
        payload_info_to_bytes,
        resolve_payload,
    )

    parser = argparse.ArgumentParser(description="DWT wavelet steganography demo")
    add_experiment_arguments(parser)
    args = parser.parse_args()

    fs = 44100
    n_samples = 65536
    n_embedded = 8192

    def _synthetic():
        rng = np.random.default_rng(0)
        t = np.arange(n_samples) / fs
        y = 0.2 * np.sin(2 * np.pi * 440 * t) + 0.05 * rng.standard_normal(n_samples)
        return y[:, np.newaxis], fs

    cover, fs, cover_path = load_cover_audio(args.cover, synthetic_builder=_synthetic)
    pinfo = resolve_payload(
        payload_file=args.payload_file,
        payload_text=args.payload_text,
        default_text="Hidden in wavelet coefficients",
    )
    payload_bytes = payload_info_to_bytes(pinfo)

    out_dir = Path(__file__).resolve().parent / "audio_out"
    stego_path = args.stego_out or (out_dir / "dwt_stego.wav")

    stego, _ = run_baseline_experiment(
        cover,
        fs,
        payload_bytes,
        n_samples,
        n_embedded,
        cover_path=cover_path,
        stego_path=stego_path,
        payload_info=pinfo,
        extracted_path=args.extracted_out,
    )
    print(f"Wrote {stego_path}")
