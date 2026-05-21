"""
Single-Kernel Echo Hiding — Python port of MATLAB 02-Echo-Hiding/01-Echo-Hiding-Single-Kernel.

Each message bit is encoded in one L-sample frame by mixing one of two echo kernels
(delay d0 for bit 0, d1 for bit 1). Decoding uses real cepstrum peak comparison at d0 vs d1.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import lfilter

# Default parameters (echo_enc_single.m / echo_dec.m)
DEFAULT_D0 = 150
DEFAULT_D1 = 200
DEFAULT_ALPHA = 0.5
DEFAULT_FRAME_LEN = 8 * 1024


# ---------------------------------------------------------------------------
# Message <-> bits
# ---------------------------------------------------------------------------

def get_bits(text: str) -> str:
    """
    Convert text to a binary string (MATLAB getBits.m).
    Each character -> 8 bits (MSB first), characters concatenated in order.
    """
    return "".join(format(ord(c), "08b") for c in text)


def bits_to_text(bits: str) -> str:
    """Decode a bit string into characters (8 bits per char, MATLAB echo_dec path)."""
    n = len(bits) // 8
    bits = bits[: n * 8]
    chars = []
    for i in range(n):
        byte = bits[i * 8 : (i + 1) * 8]
        chars.append(chr(int(byte, 2)))
    return "".join(chars)


# ---------------------------------------------------------------------------
# Mixer / windowing (mixer.m)
# ---------------------------------------------------------------------------

def _hanning_window(length: int) -> np.ndarray:
    """Periodic Hanning window matching MATLAB mixer.m / hanning()."""
    length = int(round(length))
    if length == 1:
        return np.array([1.0])
    n = np.arange(length, dtype=np.float64)
    return 0.5 * (1.0 - np.cos((2.0 * np.pi * n) / (length - 1)))


def mixer(
    frame_len: int,
    bits: str,
    lower: float = 0.0,
    upper: float = 1.0,
    smooth_len: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build block-wise and smoothed mixer signals (MATLAB mixer.m).

    Each bit is held constant over one frame (L samples), then convolved with a
    Hanning window so transitions between 0/1 regions are gradual (reduces clicks).

    Returns (w_sig, m_sig): embedding uses the smoothed w_sig (first MATLAB output).
    """
    if smooth_len <= 0 or 2 * smooth_len > frame_len:
        k = frame_len // 4
        smooth_len = k - (k % 4)
    else:
        smooth_len = smooth_len - (smooth_len % 4)

    enc = np.array([int(b) for b in bits], dtype=np.float64)
    # Each bit repeated L times: shape (num_bits * L,)
    m_sig = np.repeat(enc, frame_len)
    hann = _hanning_window(smooth_len)
    c = np.convolve(m_sig, hann, mode="full")

    # Trim to length N*L (MATLAB c(K/2+1 : end-K/2+1) with inclusive end index)
    half = smooth_len // 2
    wnorm = c[half : half + len(m_sig)]
    wnorm = wnorm / np.max(np.abs(wnorm))

    w_sig = wnorm * (upper - lower) + lower
    m_sig = m_sig * (upper - lower) + lower
    return w_sig, m_sig


# ---------------------------------------------------------------------------
# Echo kernels and embedding / extraction
# ---------------------------------------------------------------------------

def _echo_kernels(d0: int, d1: int, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Impulse trains delayed by d0/d1 samples, scaled by alpha (bit 0 / bit 1)."""
    k0 = np.zeros(d0 + 1, dtype=np.float64)
    k0[-1] = alpha
    k1 = np.zeros(d1 + 1, dtype=np.float64)
    k1[-1] = alpha
    return k0, k1


def embed_message(
    signal: np.ndarray,
    text: str,
    d0: int = DEFAULT_D0,
    d1: int = DEFAULT_D1,
    alpha: float = DEFAULT_ALPHA,
    frame_len: int = DEFAULT_FRAME_LEN,
) -> np.ndarray:
    """
    Embed text into the audio using single-kernel echo hiding (echo_enc_single.m).

    Parameters
    ----------
    signal : ndarray
        Cover audio, shape (num_samples, num_channels), float in [-1, 1] typical.
    text : str
        Message to hide.
    d0, d1 : int
        Echo delays (samples) representing bit 0 and bit 1.
    alpha : float
        Echo impulse amplitude relative to the cover.
    frame_len : int
        Samples per encoded bit (L).

    Returns
    -------
    stego : ndarray
        Stego signal (same shape as input; tail past embedded region unchanged).
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim == 1:
        signal = signal[:, np.newaxis]

    n_samples, n_ch = signal.shape
    bit_str = get_bits(text)
    n_frames = n_samples // frame_len
    # Round frame count down to a multiple of 8 (one byte per 8 frames)
    n_use = n_frames - (n_frames % 8)

    if len(bit_str) > n_use:
        warnings.warn("Message is too long, being cropped!")
        bits = bit_str[:n_use]
    else:
        warnings.warn("Message is being zero padded...")
        bits = bit_str + "0" * (n_use - len(bit_str))

    k0, k1 = _echo_kernels(d0, d1, alpha)

    # FIR echo paths: y[n] = x[n] + alpha*x[n-d]  (filter(b,1,x) in MATLAB)
    echo_zro = np.zeros_like(signal)
    echo_one = np.zeros_like(signal)
    for ch in range(n_ch):
        echo_zro[:, ch] = lfilter(k0, [1.0], signal[:, ch])
        echo_one[:, ch] = lfilter(k1, [1.0], signal[:, ch])

    # Smoothed mixer in [0, 1]: 0 -> bit0 kernel, 1 -> bit1 kernel
    window, _ = mixer(frame_len, bits, 0.0, 1.0, 256)
    mix = np.tile(window[:, np.newaxis], (1, n_ch))

    embed_len = n_use * frame_len
    cover = signal[:embed_len, :]
    mix_slice = mix[:embed_len, :]

    # mix=0: add echo_zro; mix=1: add echo_one
    stego_head = (
        cover
        + echo_zro[:embed_len, :] * np.abs(1.0 - mix_slice)
        + echo_one[:embed_len, :] * mix_slice
    )

    if embed_len < n_samples:
        out = np.vstack([stego_head, signal[embed_len:, :]])
    else:
        out = stego_head

    return out


def extract_message(
    signal: np.ndarray,
    frame_len: int = DEFAULT_FRAME_LEN,
    d0: int = DEFAULT_D0,
    d1: int = DEFAULT_D1,
    len_msg: int = 0,
) -> str:
    """
    Recover hidden text from a stego signal (echo_dec.m).

    Uses the real cepstrum per frame: compare energy at quefrency d0 vs d1.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim == 1:
        signal = signal[:, np.newaxis]

    # Decoder uses first channel only (MATLAB signal(1:N*L,1))
    mono = signal[:, 0]
    n_frames = len(mono) // frame_len
    used = n_frames * frame_len
    xsig = mono[:used].reshape(n_frames, frame_len).T  # L x N, columns = frames

    data_bits = []
    for k in range(n_frames):
        frame = xsig[:, k]
        spectrum = np.fft.fft(frame)
        # Real cepstrum: IFFT(log|FFT(x)|); phase discarded
        log_mag = np.log(np.abs(spectrum) + np.finfo(float).eps)
        rceps = np.fft.ifft(log_mag).real

        # MATLAB rceps(d0+1) vs rceps(d1+1)  ->  Python index d0, d1
        if rceps[d0] >= rceps[d1]:
            data_bits.append("0")
        else:
            data_bits.append("1")

    m = n_frames // 8
    bit_chunk = "".join(data_bits[: 8 * m])

    # MATLAB: reshape(..., 8, m)' then bin2dec — 8 bits per character, MSB first
    out = bits_to_text(bit_chunk)

    if len_msg != 0:
        out = out[:len_msg]
    return out


# ---------------------------------------------------------------------------
# Metrics (BER.m, NC.m)
# ---------------------------------------------------------------------------

def ber(hidden: str, retrieved: str) -> float:
    """
    Bit error rate in percent (BER.m).
    Compares bit strings of hidden vs retrieved text over min length.
    """
    y = get_bits(hidden)
    x = get_bits(retrieved)
    length = min(len(x), len(y))
    if length == 0:
        return 0.0
    errors = sum(1 for i in range(length) if x[i] != y[i])
    return 100.0 * (errors / length)


def nc(hidden: str, retrieved: str) -> float:
    """
    Normalized correlation between bit vectors (NC.m).
    NC = sum(x*y) / sqrt(sum(x^2) * sum(y^2))
    """
    y = get_bits(hidden)
    x = get_bits(retrieved)
    length = min(len(x), len(y))
    if length == 0:
        return 0.0
    xb = np.array([int(b) for b in x[:length]], dtype=np.float64)
    yb = np.array([int(b) for b in y[:length]], dtype=np.float64)
    s1 = np.sum(xb * yb)
    s2 = np.sqrt(np.sum(xb**2) * np.sum(yb**2))
    if s2 == 0:
        return 0.0
    return float(s1 / s2)


# ---------------------------------------------------------------------------
# I/O helpers (audioload.m / audiosave.m — WAV path only for portability)
# ---------------------------------------------------------------------------

def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Load audio as float64, shape (samples, channels)."""
    data, fs = sf.read(path, dtype="float64", always_2d=True)
    return data, fs


def save_audio(path: str | Path, data: np.ndarray, fs: int) -> None:
    """Write stego WAV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, data, fs)


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    root = (
        Path(__file__).resolve().parent.parent
        / "audio-steganography-algorithms"
        / "02-Echo-Hiding"
        / "01-Echo-Hiding-Single-Kernel"
    )
    text_path = root / "text.txt"
    message = text_path.read_text(encoding="utf-8").strip() if text_path.exists() else "Text to be hidden"

    # Synthetic cover: need at least (8 * len(message)) frames of length L
    fs = 44100
    L = DEFAULT_FRAME_LEN
    n_frames = max(8 * len(message) + 8, 200)
    n_samples = n_frames * L
    rng = np.random.default_rng(42)
    cover = rng.standard_normal((n_samples, 1))

    stego = embed_message(cover, message)
    recovered = extract_message(stego, len_msg=len(message))

    print("Original :", message)
    print("Recovered:", recovered)
    print("BER      :", ber(message, recovered))
    print("NC       :", nc(message, recovered))

    out_dir = Path(__file__).resolve().parent / "audio_out"
    save_audio(out_dir / "demo_stego.wav", stego, fs)
    print(f"Wrote {out_dir / 'demo_stego.wav'}")
