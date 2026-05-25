"""
Lightweight baseline analysis helpers for audio steganography demos.

Waveform/spectrogram plots, BER, runtime, audio quality metrics (PSNR, THD, IMD, …),
file sizes, payload loading, and markdown-style summaries.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


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
            "Byte/text recovery re-analyzes the stego waveform (lossy): expect "
            "non-zero BER unless the payload is a float audio vector matched to the band."
        ),
        "limitations": (
            "Custom detail-only cascade with truncated coefficient lengths; "
            "re-analysis after IDWT does not perfectly restore embedded coefficients; "
            "fixed db12 band; blind extract is approximate."
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
    "Autoencoder Latent": {
        "observations": (
            "Secrets are embedded as LSBs on a quantized grid in the pretrained "
            "ArchiSound latent (autoencoder1d-AT-v1, Tanh bottleneck). "
            "Waveform quality depends on how many latent scalars are flipped."
        ),
        "limitations": (
            "Lossy AE: decode(encode(cover)) != cover and encode(stego) != z_embedded; "
            "BER rises with payload size and low quant_levels. "
            "Fixed 48 kHz stereo length (default 2**18 samples)."
        ),
        "improvements": (
            "Spread-spectrum latent indexing, error-correcting codes, "
            "QIM with calibrated step size, or fine-tuned robust latent stego."
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


# ---------------------------------------------------------------------------
# Payload file loading
# ---------------------------------------------------------------------------

@dataclass
class PayloadInfo:
    """Resolved payload for embedding experiments."""

    kind: str  # "text", "binary", "audio"
    size_bytes: int
    text: str | None = None
    raw_bytes: bytes | None = None
    audio: np.ndarray | None = None
    audio_sr: int | None = None
    source_path: Path | None = None


def format_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.2f} KB"
    return f"{n / (1024 * 1024):.2f} MB"


def file_to_bytes(path: str | Path) -> bytes:
    """Read an arbitrary file as raw bytes."""
    return Path(path).read_bytes()


def bytes_to_file(data: bytes, path: str | Path) -> Path:
    """Write raw bytes to disk; create parent directories if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def bytes_to_bits(data: bytes) -> str:
    """Byte sequence to binary string (8 bits per byte, MSB first)."""
    return "".join(format(b, "08b") for b in data)


def bits_to_bytes(bits: str) -> bytes:
    """Binary string to bytes (ignores incomplete trailing bits)."""
    n = len(bits) // 8
    return bytes(int(bits[i * 8 : (i + 1) * 8], 2) for i in range(n))


def payload_to_bytes(info: PayloadInfo) -> bytes:
    """Canonical byte sequence to embed (UTF-8 for text, raw for binary files)."""
    if info.raw_bytes is not None:
        return info.raw_bytes
    if info.text is not None:
        return info.text.encode("utf-8")
    raise ValueError("PayloadInfo has no embeddable bytes")


# Clearer name for imports that must not shadow method-local helpers (e.g. DWT coeffs).
payload_info_to_bytes = payload_to_bytes


def payload_as_latin1(data: bytes) -> str:
    """Lossless bytes ↔ str bridge for text-oriented embedders (one char per byte)."""
    return data.decode("latin-1")


def bytes_from_latin1(text: str) -> bytes:
    return text.encode("latin-1")


def bit_error_rate_bytes(original: bytes, recovered: bytes) -> float:
    """BER (%) over min length in bits."""
    expected = bytes_to_bits(original)
    actual = bytes_to_bits(recovered)
    length = min(len(expected), len(actual))
    if length == 0:
        return 0.0
    errors = sum(1 for i in range(length) if expected[i] != actual[i])
    return 100.0 * errors / length


def print_ber_bytes(original: bytes, recovered: bytes) -> float:
    pct = bit_error_rate_bytes(original, recovered)
    n_bits = 8 * min(len(original), len(recovered))
    n_err = int(round(pct * n_bits / 100.0)) if n_bits else 0
    print(f"BER: {pct:.4f}%  ({n_err} bit errors / {n_bits} compared bits)")
    return pct


def suggested_extracted_path(
    method: str,
    payload_info: PayloadInfo,
    out_dir: Path,
) -> Path:
    """Choose output filename for an extracted payload."""
    if payload_info.source_path is not None:
        name = f"extracted_{payload_info.source_path.name}"
    elif payload_info.kind == "text":
        name = "extracted_payload.txt"
    else:
        name = "extracted_payload.bin"
    return out_dir / method.lower().replace(" ", "_") / name


def save_extracted_payload(
    data: bytes,
    path: Path,
    *,
    is_text_utf8: bool = False,
) -> Path:
    """Save extracted bytes; print path. Optionally verify UTF-8 text display."""
    path = bytes_to_file(data, path)
    print(f"Extracted file saved: {path}")
    if is_text_utf8:
        try:
            preview = data.decode("utf-8")
            if len(preview) > 120:
                preview = preview[:120] + "..."
            print(f"Text preview: {preview}")
        except UnicodeDecodeError:
            print("(Payload is not valid UTF-8 text; saved as binary.)")
    return path


def load_payload_file(path: str | Path) -> PayloadInfo:
    """
    Load a payload from disk.

    - .txt / .md / .json … → UTF-8 text
    - .wav / .flac / .ogg → mono float audio (for DWT-style float payloads)
    - other → raw bytes (embedded as latin-1 string in text-based methods)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    size = path.stat().st_size
    ext = path.suffix.lower()

    if ext in {".wav", ".flac", ".ogg", ".mp3", ".m4a"}:
        audio, sr = sf.read(path, dtype="float64", always_2d=False)
        audio = np.asarray(audio, dtype=np.float64)
        if audio.ndim == 2:
            audio = audio[:, 0]
        return PayloadInfo(
            kind="audio",
            size_bytes=size,
            audio=audio,
            audio_sr=int(sr),
            source_path=path,
        )

    raw = file_to_bytes(path)
    try:
        text = raw.decode("utf-8")
        if "\x00" not in text:
            return PayloadInfo(
                kind="text",
                size_bytes=size,
                text=text,
                raw_bytes=raw,
                source_path=path,
            )
    except UnicodeDecodeError:
        pass

    return PayloadInfo(
        kind="binary",
        size_bytes=size,
        text=raw.decode("latin-1"),
        raw_bytes=raw,
        source_path=path,
    )


def payload_as_text(info: PayloadInfo) -> str:
    """String for text-based embedders (LSB, echo, cepstrum)."""
    if info.text is not None:
        return info.text
    if info.raw_bytes is not None:
        return info.raw_bytes.decode("latin-1")
    raise ValueError("Payload has no text/binary content")


def add_experiment_arguments(parser: argparse.ArgumentParser) -> None:
    """CLI flags shared by stego demo scripts."""
    parser.add_argument(
        "--cover",
        type=Path,
        default=None,
        help="Cover audio file (WAV, etc.). If omitted, a synthetic cover is used.",
    )
    parser.add_argument(
        "--payload-file",
        type=Path,
        default=None,
        help="File to hide (.txt, binary, or short audio for DWT float payload).",
    )
    parser.add_argument(
        "--payload-text",
        type=str,
        default=None,
        help="Inline text payload (overrides default demo message).",
    )
    parser.add_argument(
        "--stego-out",
        type=Path,
        default=None,
        help="Output path for stego WAV (method-specific default if omitted).",
    )
    parser.add_argument(
        "--extracted-out",
        type=Path,
        default=None,
        help="Path to save extracted payload file (auto-named if omitted).",
    )


def resolve_payload(
    *,
    payload_file: Path | None = None,
    payload_text: str | None = None,
    default_text: str = "Text to be hidden",
) -> PayloadInfo:
    if payload_file is not None:
        return load_payload_file(payload_file)
    text = payload_text if payload_text is not None else default_text
    raw = text.encode("utf-8")
    return PayloadInfo(kind="text", size_bytes=len(raw), text=text, raw_bytes=raw)


def load_cover_audio(
    cover_path: Path | None,
    *,
    synthetic_builder: Callable[[], tuple[np.ndarray, int]] | None = None,
) -> tuple[np.ndarray, int, Path | None]:
    """Load cover from file or build synthetic demo signal."""
    if cover_path is not None:
        data, sr = sf.read(cover_path, dtype="float64", always_2d=True)
        return data, int(sr), cover_path
    if synthetic_builder is None:
        raise ValueError("cover_path is None and no synthetic_builder provided")
    data, sr = synthetic_builder()
    if data.ndim == 1:
        data = data[:, np.newaxis]
    return data, sr, None


# ---------------------------------------------------------------------------
# Audio quality metrics (cover vs stego) — general audio, not speech-only
# ---------------------------------------------------------------------------

def align_audio(cover: np.ndarray, stego: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Trim to common length and mono for metric computation."""
    c = mono(cover)
    s = mono(stego)
    n = min(len(c), len(s))
    return c[:n], s[:n]


def _fundamental_hz(spec: np.ndarray, freqs: np.ndarray, fmin: float = 80.0, fmax: float = 4000.0) -> float:
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return float(freqs[np.argmax(spec)])
    idx = np.argmax(spec[mask])
    return float(freqs[mask][idx])


def _thd_percent(signal: np.ndarray, sr: int, n_harmonics: int = 8) -> float:
    """
    Total harmonic distortion (%): sqrt(sum harmonic powers) / fundamental power.

    Uses the dominant peak as f0; suitable for tonal/mixed content, not a lab two-tone test.
    """
    x = np.asarray(signal, dtype=np.float64)
    n = len(x)
    if n < 256:
        return 0.0
    win = np.hanning(n)
    spec = np.abs(np.fft.rfft(x * win)) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    f0 = _fundamental_hz(spec, freqs)
    if f0 < 1.0:
        return 0.0
    i0 = int(np.argmin(np.abs(freqs - f0)))
    p0 = spec[i0]
    if p0 < 1e-20:
        return 0.0
    harmonic = 0.0
    for h in range(2, n_harmonics + 1):
        ih = int(np.argmin(np.abs(freqs - h * f0)))
        if ih < len(spec):
            harmonic += spec[ih]
    return 100.0 * np.sqrt(harmonic / p0)


def _imd_percent(signal: np.ndarray, sr: int) -> float:
    """
    Intermodulation distortion (%), simplified SMPTE-style estimate.

    Finds two strongest partials f1 < f2 and measures energy at |2f1-f2| and |2f2-f1|
    relative to (f1+f2) energy.
    """
    x = np.asarray(signal, dtype=np.float64)
    n = len(x)
    if n < 512:
        return 0.0
    win = np.hanning(n)
    spec = np.abs(np.fft.rfft(x * win)) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    mask = (freqs > 100) & (freqs < sr / 2 - 200)
    if not np.any(mask):
        return 0.0
    masked = spec.copy()
    masked[~mask] = 0
    peaks = []
    for _ in range(2):
        i = int(np.argmax(masked))
        if masked[i] < 1e-20:
            break
        peaks.append((float(freqs[i]), float(masked[i])))
        masked[max(0, i - 3) : i + 4] = 0
    if len(peaks) < 2:
        return 0.0
    peaks.sort(key=lambda p: p[0])
    f1, p1 = peaks[0]
    f2, p2 = peaks[1]
    im_freqs = [abs(2 * f1 - f2), abs(2 * f2 - f1)]
    im_power = 0.0
    for fim in im_freqs:
        if fim < 20 or fim > sr / 2 - 50:
            continue
        j = int(np.argmin(np.abs(freqs - fim)))
        im_power += spec[j]
    carrier = p1 + p2
    if carrier < 1e-20:
        return 0.0
    return 100.0 * np.sqrt(im_power / carrier)


def _lsd_db(cover: np.ndarray, stego: np.ndarray, sr: int, n_fft: int = 2048, hop: int = 512) -> float:
    """Mean log-spectral distance (dB) — common objective quality cue for any audio."""
    sc = np.abs(librosa.stft(cover, n_fft=n_fft, hop_length=hop)) + 1e-12
    ss = np.abs(librosa.stft(stego, n_fft=n_fft, hop_length=hop)) + 1e-12
    diff = np.sqrt(np.mean((20 * np.log10(sc) - 20 * np.log10(ss)) ** 2))
    return float(diff)


def correlation_coefficient(cover: np.ndarray, stego: np.ndarray) -> float:
    """
    Pearson correlation between cover and stego waveforms (mono, aligned).

    r = sum((c-mu_c)(s-mu_s)) / (N * sigma_c * sigma_s); 1.0 = identical shape/scale.
    """
    c, s = align_audio(cover, stego)
    if len(c) < 2:
        return 1.0
    c = c - np.mean(c)
    s = s - np.mean(s)
    denom = np.sqrt(np.sum(c**2) * np.sum(s**2))
    if denom < 1e-20:
        return 1.0
    return float(np.sum(c * s) / denom)


def approximate_odg(psnr_db: float, lsd_db: float) -> float:
    """
    APPROXIMATE Objective Difference Grade (ITU-R BS.1387 scale).

    True ODG requires PEAQ Model Output Variables; not available here.
    Mapping heuristic: ODG in [-4, 0], 0 = transparent, -4 = very annoying.
    Formula: ODG_approx = clip(-4, 0, -0.08*(40-PSNR) - 0.12*LSD)
    """
    odg = -0.08 * (40.0 - psnr_db) - 0.12 * lsd_db
    return float(np.clip(odg, -4.0, 0.0))


def approximate_peaq(cover: np.ndarray, stego: np.ndarray, sr: int) -> tuple[float, float]:
    """
    APPROXIMATE perceptual quality (PEAQ-inspired, not ITU-R BS.1387 compliant).

    Uses log-mel spectral RMS distance between cover and stego.
    Returns (quality_index, disturbance):
      quality_index in (0, 1], 1 = identical (higher is better)
      disturbance = mel RMS (lower is better)
    """
    c, s = align_audio(cover, stego)
    n_fft = 2048
    hop = 512
    mel_c = librosa.feature.melspectrogram(y=c, sr=sr, n_fft=n_fft, hop_length=hop)
    mel_s = librosa.feature.melspectrogram(y=s, sr=sr, n_fft=n_fft, hop_length=hop)
    diff = np.log1p(mel_c) - np.log1p(mel_s)
    disturbance = float(np.sqrt(np.mean(diff**2)))
    quality_index = float(1.0 / (1.0 + disturbance))
    return quality_index, disturbance


def compute_audio_quality_metrics(
    cover: np.ndarray,
    stego: np.ndarray,
    sr: int,
) -> dict[str, float]:
    """
    Objective cover vs stego metrics (music, speech, or effects).

    SEAQ: Spectral Error Audio Quality score (heuristic 0–100, higher = closer to cover).
    """
    c, s = align_audio(cover, stego)
    err = s - c
    mse = float(np.mean(err**2))
    peak = float(np.max(np.abs(c))) or 1.0
    psnr = 10.0 * np.log10((peak**2) / (mse + 1e-20))
    snr = 10.0 * np.log10((np.sum(c**2) + 1e-20) / (np.sum(err**2) + 1e-20))
    lsd = _lsd_db(c, s, sr)
    thd_cover = _thd_percent(c, sr)
    thd_stego = _thd_percent(s, sr)
    imd_cover = _imd_percent(c, sr)
    imd_stego = _imd_percent(s, sr)
    # Heuristic composite: PSNR (typ. 30–90 dB) and LSD (typ. 0–15 dB)
    psnr_term = min(100.0, max(0.0, (psnr - 20.0) * 1.25))
    lsd_term = min(100.0, max(0.0, 100.0 - lsd * 6.0))
    seaq = 0.5 * psnr_term + 0.5 * lsd_term
    odg = approximate_odg(psnr, lsd)
    peaq_q, peaq_dist = approximate_peaq(cover, stego, sr)
    corr = correlation_coefficient(cover, stego)

    return {
        "psnr_db": psnr,
        "snr_db": snr,
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "max_abs_diff": float(np.max(np.abs(err))),
        "lsd_db": lsd,
        "thd_cover_pct": thd_cover,
        "thd_stego_pct": thd_stego,
        "thd_delta_pct": thd_stego - thd_cover,
        "imd_cover_pct": imd_cover,
        "imd_stego_pct": imd_stego,
        "imd_delta_pct": imd_stego - imd_cover,
        "seaq_score": seaq,
        "odg_approx": odg,
        "peaq_quality_approx": peaq_q,
        "peaq_disturbance_approx": peaq_dist,
        "correlation": corr,
    }


def audio_byte_size(data: np.ndarray, subtype: str = "PCM_16") -> int:
    """Estimate on-disk PCM size for float array."""
    n = int(np.prod(data.shape))
    bps = 4 if subtype == "FLOAT" else 2
    return n * bps * (data.shape[1] if data.ndim == 2 else 1)


def _cover_size_bytes(cover_path: Path | None, cover_audio: np.ndarray | None) -> int | None:
    if cover_path and cover_path.exists():
        return cover_path.stat().st_size
    if cover_audio is not None:
        return audio_byte_size(cover_audio)
    return None


def _stego_size_bytes(stego_path: Path | None, stego_audio: np.ndarray | None) -> int | None:
    if stego_path and stego_path.exists():
        return stego_path.stat().st_size
    if stego_audio is not None:
        return audio_byte_size(stego_audio)
    return None


def print_size_report(
    phase: str,
    *,
    cover_path: Path | None = None,
    stego_path: Path | None = None,
    cover_audio: np.ndarray | None = None,
    stego_audio: np.ndarray | None = None,
    payload: PayloadInfo | None = None,
    payload_bytes: bytes | None = None,
) -> None:
    """Print cover, secret payload, and stego sizes for a given phase (before/after)."""
    print(f"\n--- Sizes ({phase}) ---")
    c_sz = _cover_size_bytes(cover_path, cover_audio)
    if c_sz is not None:
        label = str(cover_path) if cover_path else "cover (in-memory)"
        print(f"Cover audio:    {format_bytes(c_sz)}  [{label}]")

    p_sz = len(payload_bytes) if payload_bytes is not None else (payload.size_bytes if payload else None)
    if p_sz is not None:
        label = "(inline)"
        if payload and payload.source_path:
            label = str(payload.source_path)
        kind = payload.kind if payload else "bytes"
        print(f"Secret payload: {format_bytes(p_sz)}  [{kind}, {label}]")

    s_sz = _stego_size_bytes(stego_path, stego_audio)
    if s_sz is not None:
        if phase.lower().startswith("before"):
            print("Stego audio:    (not created yet)")
        else:
            label = str(stego_path) if stego_path else "stego (in-memory)"
            print(f"Stego audio:    {format_bytes(s_sz)}  [{label}]")


def print_audio_quality_metrics(metrics: dict[str, float]) -> None:
    """Print cover vs stego objective quality and similarity metrics."""
    print("\n--- Audio quality (cover vs stego) ---")
    print(f"PSNR:                    {metrics['psnr_db']:.2f} dB")
    print(f"THD (cover / stego):     {metrics['thd_cover_pct']:.4f} % / {metrics['thd_stego_pct']:.4f} %")
    print(f"IMD (cover / stego):     {metrics['imd_cover_pct']:.4f} % / {metrics['imd_stego_pct']:.4f} %")
    print(f"ODG (approximate):       {metrics['odg_approx']:.3f}  (0=transparent, -4=annoying)")
    print(f"PEAQ quality (approx):   {metrics['peaq_quality_approx']:.4f}  (1=identical)")
    print(f"PEAQ disturbance (approx): {metrics['peaq_disturbance_approx']:.4f}  (lower=better)")
    print("\n--- Recovery / similarity ---")
    print(f"Correlation coefficient: {metrics['correlation']:.6f}  (1=identical)")
    print(f"SNR:                     {metrics['snr_db']:.2f} dB")
    print(f"LSD (mean):              {metrics['lsd_db']:.2f} dB")
    print(f"SEAQ (heuristic):        {metrics['seaq_score']:.1f} / 100")


def run_full_baseline_evaluation(
    method: str,
    cover: np.ndarray,
    stego: np.ndarray,
    sr: int,
    original_bytes: bytes,
    recovered_bytes: bytes,
    embed_sec: float,
    extract_sec: float,
    plot_dir: Path,
    *,
    cover_path: Path | None = None,
    stego_path: Path | None = None,
    payload_info: PayloadInfo | None = None,
    extracted_path: Path | None = None,
    to_bits: Callable[[str], str] | None = None,
) -> dict[str, float]:
    """
    Complete baseline: sizes (before/after), BER, quality metrics, plots, save extracted file.
    """
    print_size_report(
        "before embedding",
        cover_path=cover_path,
        cover_audio=cover,
        payload=payload_info,
        payload_bytes=original_bytes,
    )
    print_size_report(
        "after embedding",
        cover_path=cover_path,
        stego_path=stego_path,
        cover_audio=cover,
        stego_audio=stego,
        payload=payload_info,
        payload_bytes=original_bytes,
    )

    print_runtime(embed_sec, extract_sec)
    print_ber_bytes(original_bytes, recovered_bytes)

    metrics = compute_audio_quality_metrics(cover, stego, sr)
    print_audio_quality_metrics(metrics)

    run_baseline_visualization(cover, stego, sr, method, plot_dir)

    if extracted_path is not None and recovered_bytes:
        is_text = payload_info is not None and payload_info.kind == "text"
        save_extracted_payload(recovered_bytes, extracted_path, is_text_utf8=is_text)

    return metrics


# Backward-compatible alias
def run_post_embed_analysis(
    cover: np.ndarray,
    stego: np.ndarray,
    sr: int,
    method: str,
    plot_dir: Path,
    *,
    cover_path: Path | None = None,
    stego_path: Path | None = None,
    payload: PayloadInfo | None = None,
) -> dict[str, float]:
    """Plots + sizes + audio quality (legacy wrapper without byte payload)."""
    run_baseline_visualization(cover, stego, sr, method, plot_dir)
    print_size_report(
        "after embedding",
        cover_path=cover_path,
        stego_path=stego_path,
        cover_audio=cover,
        stego_audio=stego,
        payload=payload,
    )
    metrics = compute_audio_quality_metrics(cover, stego, sr)
    print_audio_quality_metrics(metrics)
    return metrics
