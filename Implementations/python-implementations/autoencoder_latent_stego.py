"""
Latent-space audio steganography proof-of-concept (inference-only).

Uses ArchiSound ``autoencoder1d-AT-v1`` (pretrained, ~20M params, Tanh bottleneck).

Embedding strategy (latent LSB on a fixed quantization grid)
------------------------------------------------------------
Latent ``z`` has shape ``[B, C, T_lat]`` (default ``[1, 32, 8192]`` for 2**18 samples @ 48 kHz).

1. Flatten ``z`` in row-major C-contiguous order: index ``i`` maps to
   channel ``c = i // T_lat``, time ``t = i % T_lat``.
2. Only the first ``8 * len(payload_bytes)`` indices are modified.
3. Each scalar is clamped to ``[-1, 1]`` (Tanh bottleneck range), mapped to an integer
   ``q in [0, L-1]`` with ``L = quant_levels`` (default 2048).
4. Embed bit ``b``: force ``q' = (q & ~1) | b`` (LSB of ``q``).
5. Map ``q'`` back to float and write into ``z``.

Extraction reverses step 3–4 on ``z_rec = encode(stego_audio)`` using the same indices.

Limitations: the AE is lossy; ``encode(decode(z_embed))`` differs from ``z_embed``, so bits
are not recovered perfectly. Lower ``quant_levels`` is usually more robust; higher ``L``
packs bits more densely but is more fragile.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import soundfile as sf

# ArchiSound autoencoder1d-AT-v1 (48 kHz stereo, 32x time downsampling)
MODEL_NAME = "autoencoder1d-AT-v1"
SAMPLE_RATE = 48_000
DEFAULT_NUM_SAMPLES = 2**18  # 262144 -> latent time 8192
LATENT_CHANNELS = 32
LATENT_MIN = -1.0
LATENT_MAX = 1.0
DEFAULT_QUANT_LEVELS = 2048
METHOD_LABEL = "Autoencoder Latent"


# ---------------------------------------------------------------------------
# Latent LSB helpers
# ---------------------------------------------------------------------------


def bytes_to_bits(data: bytes) -> str:
    """Raw bytes -> binary string (MSB first per byte)."""
    return "".join(format(b, "08b") for b in data)


def bits_to_bytes(bits: str) -> bytes:
    """Binary string -> bytes (ignores trailing incomplete byte)."""
    n = len(bits) // 8
    return bytes(int(bits[i * 8 : (i + 1) * 8], 2) for i in range(n))


def latent_capacity_bits(z: np.ndarray) -> int:
    """Total scalar coefficients in latent tensor."""
    return int(z.size)


def index_to_channel_time(i: int, time_len: int) -> tuple[int, int]:
    """Flat index -> (channel, time) for [B, C, T] row-major layout."""
    return i // time_len, i % time_len


def float_to_quant(value: float, quant_levels: int) -> int:
    value = float(np.clip(value, LATENT_MIN, LATENT_MAX))
    return int(
        round((value - LATENT_MIN) / (LATENT_MAX - LATENT_MIN) * (quant_levels - 1))
    )


def quant_to_float(q: int, quant_levels: int) -> float:
    q = int(np.clip(q, 0, quant_levels - 1))
    return LATENT_MIN + (q / (quant_levels - 1)) * (LATENT_MAX - LATENT_MIN)


def embed_bit_lsb(value: float, bit: int, quant_levels: int) -> float:
    q = float_to_quant(value, quant_levels)
    q = (q & ~1) | (int(bit) & 1)
    return quant_to_float(q, quant_levels)


def extract_bit_lsb(value: float, quant_levels: int) -> int:
    q = float_to_quant(value, quant_levels)
    return q & 1


def embed_bits_in_latent(
    z: np.ndarray,
    bits: str,
    *,
    quant_levels: int = DEFAULT_QUANT_LEVELS,
    start_index: int = 0,
) -> np.ndarray:
    """
    Copy ``z`` and embed ``bits`` into flat indices ``start_index + i``.

    Modifies latent scalars in place on the returned copy.
    """
    out = np.array(z, dtype=np.float32, copy=True)
    flat = out.reshape(-1)
    t_lat = out.shape[-1]
    n = min(len(bits), flat.size - start_index)
    for i in range(n):
        flat[start_index + i] = embed_bit_lsb(
            float(flat[start_index + i]), int(bits[i]), quant_levels
        )
    return out


def extract_bits_from_latent(
    z: np.ndarray,
    n_bits: int,
    *,
    quant_levels: int = DEFAULT_QUANT_LEVELS,
    start_index: int = 0,
) -> str:
    """Read ``n_bits`` from latent using the same LSB rule."""
    flat = np.asarray(z, dtype=np.float32).reshape(-1)
    n = min(n_bits, flat.size - start_index)
    return "".join(
        str(extract_bit_lsb(float(flat[start_index + i]), quant_levels)) for i in range(n)
    )


def embed_payload_in_latent(
    z: np.ndarray,
    payload: bytes,
    *,
    quant_levels: int = DEFAULT_QUANT_LEVELS,
) -> np.ndarray:
    """Embed arbitrary bytes into the latent tensor."""
    bits = bytes_to_bits(payload)
    return embed_bits_in_latent(z, bits, quant_levels=quant_levels)


def extract_payload_from_latent(
    z: np.ndarray,
    nbytes: int,
    *,
    quant_levels: int = DEFAULT_QUANT_LEVELS,
) -> bytes:
    """Extract ``nbytes`` from latent."""
    n_bits = nbytes * 8
    bits = extract_bits_from_latent(z, n_bits, quant_levels=quant_levels)
    return bits_to_bytes(bits)[:nbytes]


def print_embedding_plan(z: np.ndarray, payload: bytes, quant_levels: int) -> None:
    """Document which latent coefficients carry the secret."""
    t_lat = z.shape[-1]
    n_bits = len(payload) * 8
    cap = latent_capacity_bits(z)
    c0, t0 = index_to_channel_time(0, t_lat)
    c1, t1 = index_to_channel_time(min(n_bits - 1, cap - 1), t_lat)
    print("\n--- Latent embedding plan ---")
    print(f"Model:              {MODEL_NAME}")
    print(f"Latent shape:       {tuple(z.shape)}  (C={z.shape[1]}, T_lat={t_lat})")
    print(f"Quant levels (L):   {quant_levels}  (LSB on integer q in [0, L-1])")
    print(f"Modified indices:   flat[0 : {n_bits}]  (channel-major flatten)")
    print(f"Index mapping:      (c, t) = (i // {t_lat}, i % {t_lat})")
    print(f"First coefficient:  index 0 -> channel {c0}, time {t0}")
    print(f"Last coefficient:   index {n_bits - 1} -> channel {c1}, time {t1}")
    print(f"Payload bytes:      {len(payload)}  ({n_bits} bits)")
    print(f"Latent capacity:    {cap} scalars ({cap // 8} bytes if all used)")
    print(
        "Rule: clamp v in [-1,1] -> q=round(v); q=(q&~1)|bit; v'=dequantize(q')"
    )


# ---------------------------------------------------------------------------
# ArchiSound I/O
# ---------------------------------------------------------------------------

# Hugging Face revision pinned by archisound for autoencoder1d-AT-v1
ARCHISOUND_REVISION = "57b6cde1969208d10fdd3e813708c1abe49f25c1"


def _ensure_hf_pretrained_compat() -> None:
    """
    Transformers 5.x finalizes loading with ``model.all_tied_weights_keys``.

    ArchiSound's remote AutoEncoder1d predates that API (no ``post_init()``).
    Patch once so ``from_pretrained`` succeeds on transformers>=5; pinning <5 is
    still recommended (see autoencoder_latent_requirements.txt).
    """
    import transformers
    from transformers import PreTrainedModel

    major = int(transformers.__version__.split(".")[0])
    if major < 5:
        return
    if getattr(PreTrainedModel, "_srip_archisound_compat", False):
        return

    _orig = PreTrainedModel._move_missing_keys_from_meta_to_device

    def _patched_move_missing(self, *args, **kwargs):
        if not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys = {}
        return _orig(self, *args, **kwargs)

    PreTrainedModel._move_missing_keys_from_meta_to_device = _patched_move_missing
    PreTrainedModel._srip_archisound_compat = True


def _finalize_loaded_model(model) -> None:
    """Ensure tied-weights dict exists for later transformers 5.x helpers."""
    if not hasattr(model, "all_tied_weights_keys"):
        model.all_tied_weights_keys = {}


def load_autoencoder(device: str | None = None):
    """Load pretrained ArchiSound autoencoder (eval mode)."""
    try:
        from archisound import ArchiSound
    except ImportError as e:
        raise ImportError(
            "archisound is required for latent stego. Install with:\n"
            "  pip install -r autoencoder_latent_requirements.txt"
        ) from e

    import torch

    _ensure_hf_pretrained_compat()
    ae = ArchiSound.from_pretrained(MODEL_NAME)
    _finalize_loaded_model(ae)
    ae.eval()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    ae = ae.to(device)
    return ae, device


def _tensor_encode(ae, x) -> object:
    out = ae.encode(x)
    return out[0] if isinstance(out, tuple) else out


def _tensor_decode(ae, z) -> object:
    out = ae.decode(z)
    return out[0] if isinstance(out, tuple) else out


def numpy_to_model_input(
    audio: np.ndarray,
    *,
    num_samples: int = DEFAULT_NUM_SAMPLES,
) -> np.ndarray:
    """
    Prepare float32 numpy audio as stereo ``[1, 2, T]`` for the AE.

    Mono is duplicated to two channels. Truncates or zero-pads to ``num_samples``.
    """
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim == 2:
        if audio.shape[1] >= 2:
            left, right = audio[:, 0], audio[:, 1]
        else:
            left = right = audio[:, 0]
    else:
        left = right = audio

    n = min(len(left), len(right), num_samples)
    left = left[:n]
    right = right[:n]
    if n < num_samples:
        left = np.pad(left, (0, num_samples - n))
        right = np.pad(right, (0, num_samples - n))

    stereo = np.stack([left, right], axis=0)
    return stereo[np.newaxis, :, :].astype(np.float32)


def model_output_to_numpy(y) -> np.ndarray:
    """``[1, 2, T]`` tensor/array -> mono float64 for analysis/plots."""
    import torch

    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 3:
        y = y[0]
    if y.ndim == 2 and y.shape[0] >= 2:
        return 0.5 * (y[0] + y[1])
    return y.reshape(-1)


def prepare_cover_audio(
    audio: np.ndarray,
    sr: int,
    *,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    target_sr: int = SAMPLE_RATE,
) -> tuple[np.ndarray, int]:
    """Resample (if needed) and format cover for the AE."""
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]

    if sr != target_sr:
        try:
            import librosa

            audio = librosa.resample(
                audio.T, orig_sr=sr, target_sr=target_sr
            ).T
            if audio.ndim == 1:
                audio = audio[:, np.newaxis]
            sr = target_sr
        except Exception as e:
            warnings.warn(f"librosa resample failed ({e}); using original sr={sr}")

    model_in = numpy_to_model_input(audio, num_samples=num_samples)
    return model_in, sr


def latent_to_numpy(z) -> np.ndarray:
    import torch

    if isinstance(z, torch.Tensor):
        return z.detach().cpu().numpy()
    return np.asarray(z, dtype=np.float32)


# ---------------------------------------------------------------------------
# Stego pipeline
# ---------------------------------------------------------------------------


def embed_latent_stego(
    ae,
    cover_model_in: np.ndarray,
    payload: bytes,
    *,
    quant_levels: int = DEFAULT_QUANT_LEVELS,
    device: str = "cpu",
) -> tuple[object, np.ndarray, np.ndarray]:
    """
    cover -> encode -> embed in z -> decode -> stego waveform tensor.

    Returns (stego_tensor, z_cover_np, z_stego_np).
    """
    import torch

    x = torch.from_numpy(cover_model_in).to(device)
    with torch.no_grad():
        z = _tensor_encode(ae, x)
    z_np = latent_to_numpy(z)
    print_embedding_plan(z_np, payload, quant_levels)
    z_emb_np = embed_payload_in_latent(z_np, payload, quant_levels=quant_levels)
    z_emb = torch.from_numpy(z_emb_np).to(device=device, dtype=z.dtype)
    z_emb = z_emb.reshape(z.shape)
    with torch.no_grad():
        stego = _tensor_decode(ae, z_emb)
    return stego, z_np, z_emb_np


def extract_latent_payload(
    ae,
    stego_model_in: np.ndarray,
    nbytes: int,
    *,
    quant_levels: int = DEFAULT_QUANT_LEVELS,
    device: str = "cpu",
) -> tuple[bytes, np.ndarray]:
    """stego audio -> encode -> extract bytes from latent."""
    import torch

    x = torch.from_numpy(stego_model_in).to(device)
    with torch.no_grad():
        z = _tensor_encode(ae, x)
    z_np = latent_to_numpy(z)
    return extract_payload_from_latent(z_np, nbytes, quant_levels=quant_levels), z_np


# ---------------------------------------------------------------------------
# Baseline experiment + CLI
# ---------------------------------------------------------------------------


def run_baseline_experiment(
    cover_audio: np.ndarray,
    sr: int,
    payload_bytes: bytes,
    *,
    quant_levels: int = DEFAULT_QUANT_LEVELS,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    plot_dir: Path | None = None,
    cover_path: Path | None = None,
    stego_path: Path | None = None,
    payload_info: object | None = None,
    extracted_path: Path | None = None,
    device: str | None = None,
) -> tuple[np.ndarray, bytes]:
    """Full latent embed/extract with stego_analysis evaluation."""
    from stego_analysis import (
        PayloadInfo,
        print_analysis_report,
        resolve_payload,
        run_full_baseline_evaluation,
        suggested_extracted_path,
        time_embed_extract,
    )

    plot_dir = plot_dir or (
        Path(__file__).resolve().parent / "audio_out" / "analysis" / "autoencoder"
    )
    pinfo = (
        payload_info
        if isinstance(payload_info, PayloadInfo)
        else resolve_payload(payload_text=payload_bytes.decode("utf-8", errors="replace"))
    )

    ae, device = load_autoencoder(device)
    cover_in, sr = prepare_cover_audio(
        cover_audio, sr, num_samples=num_samples, target_sr=SAMPLE_RATE
    )
    state: dict = {}

    def _embed():
        stego_t, _, _ = embed_latent_stego(
            ae,
            cover_in,
            payload_bytes,
            quant_levels=quant_levels,
            device=device,
        )
        state["stego_t"] = stego_t
        state["stego_np"] = model_output_to_numpy(stego_t)
        state["cover_np"] = 0.5 * (cover_in[0, 0] + cover_in[0, 1])
        return stego_t

    def _extract():
        stego_in = numpy_to_model_input(state["stego_np"], num_samples=num_samples)
        recovered, _ = extract_latent_payload(
            ae,
            stego_in,
            len(payload_bytes),
            quant_levels=quant_levels,
            device=device,
        )
        return recovered

    embed_sec, extract_sec, _, recovered = time_embed_extract(_embed, _extract)

    stego_np = state["stego_np"]
    cover_np = state["cover_np"]

    if stego_path is not None:
        stego_path = Path(stego_path)
        stego_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(stego_path, stego_np, SAMPLE_RATE)

    out_extract = extracted_path or suggested_extracted_path(
        METHOD_LABEL,
        pinfo,
        Path(__file__).resolve().parent / "audio_out" / "extracted",
    )

    print(f"\n=== {METHOD_LABEL} baseline experiment ===")
    run_full_baseline_evaluation(
        METHOD_LABEL,
        cover_np[:, np.newaxis],
        stego_np[:, np.newaxis],
        SAMPLE_RATE,
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
    print_analysis_report(METHOD_LABEL)
    return stego_np, recovered


if __name__ == "__main__":
    import argparse

    from stego_analysis import (
        add_experiment_arguments,
        load_cover_audio,
        payload_info_to_bytes,
        resolve_payload,
    )

    parser = argparse.ArgumentParser(
        description="Latent-space audio steganography (ArchiSound autoencoder1d-AT-v1)"
    )
    add_experiment_arguments(parser)
    parser.add_argument(
        "--quant-levels",
        type=int,
        default=DEFAULT_QUANT_LEVELS,
        help="Latent quantization levels for LSB embedding (lower=more robust).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help="Cover length in samples (default 2**18 for AT-v1).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="torch device (default: cuda if available else cpu).",
    )
    args = parser.parse_args()

    def _synthetic():
        t = np.arange(args.num_samples) / SAMPLE_RATE
        tone = 0.35 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.sin(2 * np.pi * 880 * t)
        return tone[:, np.newaxis], SAMPLE_RATE

    cover, sr, cover_path = load_cover_audio(args.cover, synthetic_builder=_synthetic)
    pinfo = resolve_payload(
        payload_file=args.payload_file,
        payload_text=args.payload_text,
        default_text="Hidden in ArchiSound latent",
    )
    payload_bytes = payload_info_to_bytes(pinfo)

    out_dir = Path(__file__).resolve().parent / "audio_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    stego_path = args.stego_out or (out_dir / "autoencoder_latent_stego.wav")

    try:
        stego, _ = run_baseline_experiment(
            cover,
            sr,
            payload_bytes,
            quant_levels=args.quant_levels,
            num_samples=args.num_samples,
            cover_path=cover_path,
            stego_path=stego_path,
            payload_info=pinfo,
            extracted_path=args.extracted_out,
            device=args.device,
        )
        print(f"Wrote {stego_path}")
    except ImportError as err:
        print(f"ERROR: {err}")
        raise SystemExit(1) from err
