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
    """Hide UTF-8 text in a 16-bit WAV (stored as bytes via embed_bytes)."""
    embed_bytes(wavin, wavout, text.encode("utf-8"), password)


def extract_message(
    wavin: str | Path,
    password: str = "password123",
) -> str:
    """Recover hidden text from a stego WAV; returns empty string if password/check fails."""
    data = extract_bytes(wavin, password)
    return data.decode("utf-8", errors="replace")


def embed_bytes(
    wavin: str | Path,
    wavout: str | Path,
    data: bytes,
    password: str = "password123",
) -> None:
    """Hide raw bytes in a 16-bit WAV (length field = byte count)."""
    wavin, wavout = Path(wavin), Path(wavout)
    header, dsize, cover = _read_wav_pcm(wavin)

    if not data:
        raise ValueError("Payload is empty")

    bin_arr = _decimal_to_bits(np.array(list(data), dtype=np.float64), 8)
    m, n = bin_arr.shape
    len_msg = m * n
    length_bits = _decimal_to_bits(np.array([m], dtype=np.float64), 40).reshape(1, -1)

    bitx = np.bitwise_xor(bin_arr.ravel(order="F"), _prng(password, len_msg))
    binx = bitx.reshape(m, n, order="F")

    if cover.size < len_msg + 48:
        raise ValueError("Message is too long!")

    control = _decimal_to_bits(
        np.array([sum(ord(c) for c in password) % 256], dtype=np.float64), 8
    ).reshape(-1)

    _set_lsb(cover[0:8], control)
    _set_lsb(cover[8:48], length_bits.ravel())
    _set_lsb(cover[48 : 48 + len_msg], binx.ravel(order="F"))

    _write_wav_pcm(wavout, header, dsize, cover)


def extract_bytes(
    wavin: str | Path,
    password: str = "password123",
) -> bytes:
    """Recover hidden bytes from a stego WAV."""
    wavin = Path(wavin)
    _, _, stego = _read_wav_pcm(wavin)

    control = _get_lsb(stego[0:8])
    expected = sum(ord(c) for c in password) % 256
    if int(_bits_to_decimal(control.reshape(1, -1))[0]) != expected:
        warnings.warn("Password is wrong or message is corrupted!")
        return b""

    length_bits = _get_lsb(stego[8:48]).reshape(1, -1)
    num_bytes = int(_bits_to_decimal(length_bits)[0])
    len_bits = num_bytes * 8

    raw_bits = _get_lsb(stego[48 : 48 + len_bits])
    dat = np.bitwise_xor(raw_bits, _prng(password, len_bits))
    bin_arr = dat.reshape(num_bytes, 8, order="F")
    return bytes(_bits_to_decimal(bin_arr).astype(np.uint8))


# ---------------------------------------------------------------------------
# Example usage + baseline experiment
# ---------------------------------------------------------------------------

def run_baseline_experiment(
    cover_path: Path,
    stego_path: Path,
    payload_bytes: bytes,
    password: str = "mypassword123",
    plot_dir: Path | None = None,
    payload_info: object | None = None,
    extracted_path: Path | None = None,
) -> None:
    """Embed/extract with full evaluation framework."""
    import soundfile as sf

    from stego_analysis import (
        PayloadInfo,
        print_analysis_report,
        resolve_payload,
        run_full_baseline_evaluation,
        suggested_extracted_path,
        time_embed_extract,
    )

    plot_dir = plot_dir or Path(__file__).resolve().parent / "audio_out" / "analysis" / "lsb"
    pinfo = (
        payload_info
        if isinstance(payload_info, PayloadInfo)
        else resolve_payload(payload_text=payload_bytes.decode("utf-8", errors="replace"))
    )
    out_extract = extracted_path or suggested_extracted_path(
        "LSB", pinfo, Path(__file__).resolve().parent / "audio_out" / "extracted"
    )

    embed_sec, extract_sec, _, recovered = time_embed_extract(
        lambda: embed_bytes(cover_path, stego_path, payload_bytes, password),
        lambda: extract_bytes(stego_path, password),
    )

    cover_f, sr = sf.read(cover_path, dtype="float64", always_2d=True)
    stego_f, _ = sf.read(stego_path, dtype="float64", always_2d=True)

    print("\n=== LSB baseline experiment ===")
    run_full_baseline_evaluation(
        "LSB",
        cover_f,
        stego_f,
        sr,
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
    print_analysis_report("LSB")


if __name__ == "__main__":
    import argparse

    import soundfile as sf

    from stego_analysis import add_experiment_arguments, payload_to_bytes, resolve_payload

    parser = argparse.ArgumentParser(description="LSB audio steganography demo")
    add_experiment_arguments(parser)
    parser.add_argument("--password", default="mypassword123")
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent / "audio_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    cover_path = args.cover or (out_dir / "lsb_cover.wav")
    stego_path = args.stego_out or (out_dir / "lsb_stego.wav")

    if args.cover is None:
        fs = 44100
        duration = 1.0
        t = np.arange(int(fs * duration)) / fs
        tone = (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        sf.write(cover_path, tone, fs, subtype="PCM_16")

    pinfo = resolve_payload(payload_file=args.payload_file, payload_text=args.payload_text)
    payload_bytes = payload_to_bytes(pinfo)

    run_baseline_experiment(
        cover_path,
        stego_path,
        payload_bytes,
        args.password,
        payload_info=pinfo,
        extracted_path=args.extracted_out,
    )
