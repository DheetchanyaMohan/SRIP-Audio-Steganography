"""
LSB audio steganography — Python port of MATLAB 03-LSB-Coding (lsb_enc.m / lsb_dec.m).

Embeds one bit per 16-bit PCM sample (LSB only), with a password-derived PRNG XOR layer.
WAV layout matches the original: 40-byte header chunk, 32-bit data-size field, uint16 PCM.
"""

from __future__ import annotations

import struct
import warnings
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# WAV I/O (binary layout identical to MATLAB fopen/fread/fwrite)
# ---------------------------------------------------------------------------

def _read_wav_pcm(path: Path) -> tuple[bytes, int, np.ndarray]:
    """Load cover/stego: header (40 B), data chunk size (uint32), PCM uint16 samples."""
    raw = path.read_bytes()
    header = raw[:40]
    (dsize,) = struct.unpack_from("<I", raw, 40)
    pcm = np.frombuffer(raw[44:], dtype=np.uint16).copy()
    return header, dsize, pcm


def _write_wav_pcm(path: Path, header: bytes, dsize: int, pcm: np.ndarray) -> None:
    """Write stego WAV preserving the original RIFF header bytes."""
    path.write_bytes(header + struct.pack("<I", dsize) + pcm.astype(np.uint16).tobytes())


# ---------------------------------------------------------------------------
# Bit helpers (MATLAB d2b / b2d, LSB in column 0)
# ---------------------------------------------------------------------------

def _decimal_to_bits(values: np.ndarray, n_bits: int) -> np.ndarray:
    """
    Minimal de2bi: each value -> n_bits columns, powers 2^0 .. 2^(n_bits-1).
    Matches MATLAB d2b() column order.
    """
    d = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    power = (2.0 ** np.arange(n_bits)).reshape(1, -1)
    return np.floor(np.remainder(d, 2 * power) / power).astype(np.uint8)


def _bits_to_decimal(bits: np.ndarray) -> np.ndarray:
    """Minimal bi2de: sum of bit_k * 2^k along columns (MATLAB b2d)."""
    if bits.size == 0:
        return np.array([], dtype=np.float64)
    b = np.asarray(bits, dtype=np.float64)
    weights = 2.0 ** np.arange(b.shape[1])
    return (b @ weights).reshape(-1)


# ---------------------------------------------------------------------------
# PRNG (password-scrambling; see module docstring on MATLAB vs NumPy)
# ---------------------------------------------------------------------------

def _prng(key: str, length: int) -> np.ndarray:
    """
    Pseudorandom binary stream for XOR with payload bits.
    MATLAB: rand('seed', sum(double(key).*(1:length(key)))); rand(L,1)>0.5
    """
    seed = sum(ord(c) * (i + 1) for i, c in enumerate(key))
    rng = np.random.RandomState(seed)
    return (rng.rand(length) > 0.5).astype(np.uint8)


# ---------------------------------------------------------------------------
# Sample LSB read/write (MATLAB bitget/bitset bit position 1 = LSB)
# ---------------------------------------------------------------------------

def _set_lsb(samples: np.ndarray, bits: np.ndarray) -> None:
    """Force each sample's least significant bit to the corresponding value in bits."""
    bits = np.asarray(bits, dtype=np.uint16).ravel()
    samples[:] = (samples & np.uint16(0xFFFE)) | bits


def _get_lsb(samples: np.ndarray) -> np.ndarray:
    """Read LSB of each sample (bit 1 in MATLAB terminology)."""
    return (samples & 1).astype(np.uint8)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed_message(
    wavin: str | Path,
    wavout: str | Path,
    text: str,
    password: str = "password123",
) -> None:
    """
    Hide text in a 16-bit WAV using LSB substitution + password XOR.

    Layout in PCM samples (0-based indices):
      [0:8]   control byte (mod 256 of sum of password ASCII)
      [8:48]  40-bit message length (character count)
      [48:48+len_msg]  scrambled message bits
    """
    wavin, wavout = Path(wavin), Path(wavout)
    header, dsize, cover = _read_wav_pcm(wavin)

    # Message -> binary matrix (rows = characters, cols = 8 bits), column-major flatten
    bin_arr = _decimal_to_bits(np.array([ord(c) for c in text], dtype=np.float64), 8)
    m, n = bin_arr.shape
    len_msg = m * n
    length_bits = _decimal_to_bits(np.array([m], dtype=np.float64), 40).reshape(1, -1)

    bitx = np.bitwise_xor(bin_arr.ravel(order="F"), _prng(password, len_msg))
    binx = bitx.reshape(m, n, order="F")

    if cover.size < len_msg + 48:
        raise ValueError("Message is too long!")

    # Control field: checksum derived from password
    control = _decimal_to_bits(
        np.array([sum(ord(c) for c in password) % 256], dtype=np.float64), 8
    ).reshape(-1)

    _set_lsb(cover[0:8], control)
    _set_lsb(cover[8:48], length_bits.ravel())
    _set_lsb(cover[48 : 48 + len_msg], binx.ravel(order="F"))

    _write_wav_pcm(wavout, header, dsize, cover)


def extract_message(
    wavin: str | Path,
    password: str = "password123",
) -> str:
    """Recover hidden text from a stego WAV; returns empty string if password/check fails."""
    wavin = Path(wavin)
    _, _, stego = _read_wav_pcm(wavin)

    control = _get_lsb(stego[0:8])
    expected = sum(ord(c) for c in password) % 256
    if int(_bits_to_decimal(control.reshape(1, -1))[0]) != expected:
        warnings.warn("Password is wrong or message is corrupted!")
        return ""

    length_bits = _get_lsb(stego[8:48]).reshape(1, -1)
    num_chars = int(_bits_to_decimal(length_bits)[0])
    len_bits = num_chars * 8

    raw_bits = _get_lsb(stego[48 : 48 + len_bits])
    dat = np.bitwise_xor(raw_bits, _prng(password, len_bits))
    bin_arr = dat.reshape(num_chars, 8, order="F")
    chars = _bits_to_decimal(bin_arr).astype(np.uint8)
    return "".join(chr(c) for c in chars)


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import soundfile as sf

    # Paths relative to this script; adjust to your cover WAV
    root = Path(__file__).resolve().parent.parent / "audio-steganography-algorithms" / "03-LSB-Coding"
    cover = root / "cover.wav"  # provide a 16-bit mono/stereo WAV
    stego = root / "cover_stego.wav"
    password = "mypassword123"
    message = "Text to be hidden"

    if cover.exists():
        embed_message(cover, stego, message, password)
        recovered = extract_message(stego, password)
        print("Retrieved message:", recovered)

        # Optional: load stego with soundfile for playback/analysis (float [-1, 1])
        audio, sr = sf.read(stego, dtype="float32")
        print(f"Stego audio: {len(audio)} samples @ {sr} Hz")
    else:
        print(f"Place a 16-bit WAV at {cover} to run the example.")
