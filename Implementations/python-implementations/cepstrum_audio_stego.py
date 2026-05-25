"""
Cepstrum-domain audio steganography (educational).

Each message bit is stored in one frame by nudging a chosen quefrency coefficient
of the real cepstrum (homomorphic "log-spectrum" domain). Phase from the original
frame is reused on synthesis so the sound stays close to the cover.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import soundfile as sf

# Small constant avoids log(0) when a spectral bin is silent
_LOG_EPS = 1e-12

# Default embedding: quefrency index and frame size (samples)
DEFAULT_FRAME_LEN = 2048
DEFAULT_QUEFRENCY = 40
DEFAULT_STRENGTH = 0.05


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Load WAV as float64, shape (num_samples, num_channels)."""
    data, fs = sf.read(path, dtype="float64", always_2d=True)
    return data, fs


def save_audio(path: str | Path, data: np.ndarray, fs: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, data, fs)


# ---------------------------------------------------------------------------
# Message <-> bits
# ---------------------------------------------------------------------------

def bytes_to_bits(data: bytes) -> str:
    """Raw bytes -> binary string (8 bits per byte, MSB first)."""
    return "".join(format(b, "08b") for b in data)


def bits_to_bytes(bits: str) -> bytes:
    """Binary string -> bytes (ignores incomplete trailing bits)."""
    n = len(bits) // 8
    return bytes(int(bits[i * 8 : (i + 1) * 8], 2) for i in range(n))


def text_to_bits(text: str) -> str:
    """UTF-8 text -> binary string (8 bits per byte, MSB first)."""
    return bytes_to_bits(text.encode("utf-8"))


def bits_to_text(bits: str) -> str:
    """Binary string -> UTF-8 text (truncates to full bytes only)."""
    return bits_to_bytes(bits).decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Cepstrum analysis / synthesis
# ---------------------------------------------------------------------------

def compute_real_cepstrum(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    One-frame homomorphic analysis chain.

    Steps
    -----
    1. FFT  — time frame -> complex spectrum X[k]
    2. Log-magnitude — log|X[k]| compresses dynamic range (additive structure)
    3. Real cepstrum — IFFT(log|X|); result is real when log-mag is symmetric
    """
    spectrum = np.fft.fft(frame)
    log_magnitude = np.log(np.abs(spectrum) + _LOG_EPS)
    # Real cepstrum: discard phase of log-spectrum (common "real cepstrum" definition)
    cepstrum = np.fft.ifft(log_magnitude).real
    return spectrum, log_magnitude, cepstrum


def reconstruct_frame_from_cepstrum(
    cepstrum: np.ndarray,
    original_phase: np.ndarray,
    frame_len: int,
) -> np.ndarray:
    """
    Inverse homomorphic synthesis (approximate).

    1. FFT(cepstrum) -> estimate of log-magnitude spectrum
    2. exp(.) -> magnitude spectrum
    3. Reuse original FFT phase (keeps timbre stable; only magnitude is warped)
    4. IFFT -> time-domain frame
    """
    log_mag_hat = np.fft.fft(cepstrum).real
    magnitude = np.exp(np.clip(log_mag_hat, -50.0, 50.0))
    spectrum = magnitude * np.exp(1j * original_phase)
    return np.fft.ifft(spectrum).real[:frame_len]


def _valid_quefrency(index: int, frame_len: int) -> int:
    """Keep index in causal quefrency region [1, frame_len//2 - 1]."""
    return int(np.clip(index, 1, frame_len // 2 - 1))


# ---------------------------------------------------------------------------
# Bit embed / extract in cepstrum
# ---------------------------------------------------------------------------

def embed_bit_in_cepstrum(
    cepstrum: np.ndarray,
    bit: int,
    quefrency: int,
    strength: float,
) -> np.ndarray:
    """
    Nudge one cepstral coefficient to encode bit 0 or 1.

    bit 0 -> subtract strength
    bit 1 -> add strength

    Symmetric pair (q, N-q) is updated to preserve real-valued log-mag symmetry.
    """
    out = cepstrum.copy()
    q = _valid_quefrency(quefrency, len(cepstrum))
    delta = strength if bit else -strength
    out[q] += delta
    out[-q] += delta  # mirror for conjugate-symmetric log-spectrum
    return out


def extract_bit_from_cepstrum(
    cepstrum: np.ndarray,
    quefrency: int,
) -> int:
    """Decode bit by sign of the selected cepstral coefficient (threshold 0)."""
    q = _valid_quefrency(quefrency, len(cepstrum))
    return 1 if cepstrum[q] >= 0.0 else 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed_bytes(
    cover: np.ndarray,
    data: bytes,
    frame_len: int = DEFAULT_FRAME_LEN,
    quefrency: int = DEFAULT_QUEFRENCY,
    strength: float = DEFAULT_STRENGTH,
) -> np.ndarray:
    """
    Hide raw bytes in non-overlapping frames of the cover audio.

    Capacity: one bit per frame -> roughly (len(cover) / frame_len) bits.
    """
    if cover.ndim == 2:
        mono = cover[:, 0].astype(np.float64).copy()
        n_ch = cover.shape[1]
    else:
        mono = np.asarray(cover, dtype=np.float64).copy()
        n_ch = 1

    bits = bytes_to_bits(data)
    n_frames_avail = len(mono) // frame_len
    if len(bits) > n_frames_avail:
        warnings.warn("Message truncated to fit available frames.")
        bits = bits[:n_frames_avail]
    else:
        bits = bits + "0" * (n_frames_avail - len(bits))

    stego = mono.copy()
    for i, bit_char in enumerate(bits):
        start = i * frame_len
        frame = stego[start : start + frame_len]
        spectrum, _, cepstrum = compute_real_cepstrum(frame)
        bit = 1 if bit_char == "1" else 0
        cep_mod = embed_bit_in_cepstrum(cepstrum, bit, quefrency, strength)
        phase = np.angle(spectrum)
        stego[start : start + frame_len] = reconstruct_frame_from_cepstrum(
            cep_mod, phase, frame_len
        )

    if n_ch > 1:
        out = cover.astype(np.float64).copy()
        out[: len(stego), 0] = stego
        return out
    return stego


def embed_message(
    cover: np.ndarray,
    text: str,
    frame_len: int = DEFAULT_FRAME_LEN,
    quefrency: int = DEFAULT_QUEFRENCY,
    strength: float = DEFAULT_STRENGTH,
) -> np.ndarray:
    """Hide UTF-8 text (convenience wrapper around embed_bytes)."""
    return embed_bytes(cover, text.encode("utf-8"), frame_len, quefrency, strength)


def extract_bytes(
    stego: np.ndarray,
    frame_len: int = DEFAULT_FRAME_LEN,
    quefrency: int = DEFAULT_QUEFRENCY,
    n_bits: int | None = None,
) -> bytes:
    """Recover raw bytes from cepstral coefficients."""
    if stego.ndim == 2:
        mono = stego[:, 0].astype(np.float64)
    else:
        mono = np.asarray(stego, dtype=np.float64)

    n_frames = len(mono) // frame_len
    count = n_frames if n_bits is None else min(n_bits, n_frames)

    bits = []
    for i in range(count):
        frame = mono[i * frame_len : (i + 1) * frame_len]
        _, _, cepstrum = compute_real_cepstrum(frame)
        bits.append(str(extract_bit_from_cepstrum(cepstrum, quefrency)))

    return bits_to_bytes("".join(bits))


def extract_message(
    stego: np.ndarray,
    frame_len: int = DEFAULT_FRAME_LEN,
    quefrency: int = DEFAULT_QUEFRENCY,
    n_bits: int | None = None,
) -> str:
    """Recover bits from cepstral coefficients and decode UTF-8 text."""
    data = extract_bytes(stego, frame_len, quefrency, n_bits=n_bits)
    return data.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Example usage + baseline experiment
# ---------------------------------------------------------------------------

def run_baseline_experiment(
    cover: np.ndarray,
    fs: int,
    payload_bytes: bytes,
    frame_len: int = DEFAULT_FRAME_LEN,
    quefrency: int = DEFAULT_QUEFRENCY,
    strength: float = DEFAULT_STRENGTH,
    plot_dir: Path | None = None,
    *,
    cover_path: Path | None = None,
    stego_path: Path | None = None,
    payload_info: object | None = None,
    extracted_path: Path | None = None,
) -> tuple[np.ndarray, bytes]:
    """Embed/extract with full evaluation framework (same baseline as LSB/Echo)."""
    from stego_analysis import (
        PayloadInfo,
        print_analysis_report,
        resolve_payload,
        run_full_baseline_evaluation,
        suggested_extracted_path,
        time_embed_extract,
    )

    plot_dir = plot_dir or Path(__file__).resolve().parent / "audio_out" / "analysis" / "cepstrum"
    pinfo = (
        payload_info
        if isinstance(payload_info, PayloadInfo)
        else resolve_payload(payload_text=payload_bytes.decode("utf-8", errors="replace"))
    )
    n_bits = len(bytes_to_bits(payload_bytes))
    state: dict = {}

    def _embed():
        stego = embed_bytes(cover.copy(), payload_bytes, frame_len, quefrency, strength)
        state["stego"] = stego
        return stego

    def _extract():
        return extract_bytes(state["stego"], frame_len, quefrency, n_bits=n_bits)

    embed_sec, extract_sec, _, recovered = time_embed_extract(_embed, _extract)
    stego = state["stego"]

    if stego_path is not None:
        save_audio(stego_path, stego, fs)

    out_extract = extracted_path or suggested_extracted_path(
        "Cepstrum", pinfo, Path(__file__).resolve().parent / "audio_out" / "extracted"
    )

    print("\n=== Cepstrum baseline experiment ===")
    print(f"Frames used: {n_bits} (frame_len={frame_len})")
    run_full_baseline_evaluation(
        "Cepstrum",
        cover,
        stego,
        fs,
        payload_bytes,
        recovered,
        embed_sec,
        extract_sec,
        plot_dir,
        cover_path=cover_path,
        stego_path=stego_path,
        payload_info=pinfo,
        extracted_path=out_extract,
    )
    print_analysis_report("Cepstrum")
    return stego, recovered


if __name__ == "__main__":
    import argparse

    from stego_analysis import (
        add_experiment_arguments,
        load_cover_audio,
        payload_info_to_bytes,
        resolve_payload,
    )

    parser = argparse.ArgumentParser(description="Cepstrum steganography demo")
    add_experiment_arguments(parser)
    args = parser.parse_args()

    fs = 44100

    def _synthetic():
        duration = 6.0
        n = int(fs * duration)
        t = np.arange(n) / fs
        rng = np.random.default_rng(0)
        y = (
            0.4 * np.sin(2 * np.pi * 440 * t)
            + 0.1 * np.sin(2 * np.pi * 880 * t)
            + 0.02 * rng.standard_normal(n)
        )
        return y[:, np.newaxis], fs

    cover, fs, cover_path = load_cover_audio(args.cover, synthetic_builder=_synthetic)
    pinfo = resolve_payload(
        payload_file=args.payload_file,
        payload_text=args.payload_text,
        default_text="Hello cepstrum",
    )
    payload_bytes = payload_info_to_bytes(pinfo)

    out_dir = Path(__file__).resolve().parent / "audio_out"
    stego_path = args.stego_out or (out_dir / "cepstrum_stego.wav")

    stego, _ = run_baseline_experiment(
        cover,
        fs,
        payload_bytes,
        cover_path=cover_path,
        stego_path=stego_path,
        payload_info=pinfo,
        extracted_path=args.extracted_out,
    )
    print(f"Wrote {stego_path}")
