"""
Latent-space audio steganography proof-of-concept (inference-only).

Uses ArchiSound ``autoencoder1d-AT-v1`` (pretrained, ~20M params, Tanh bottleneck).

Embedding strategy (coarse-grid LSB + 5x repetition)
----------------------------------------------------
Latent ``z`` has shape ``[B, C, T_lat]`` (default ``[1, 32, 8192]`` for 2**18 samples @ 48 kHz).

1. Flatten ``z`` in row-major C-contiguous order: index ``i`` maps to
   channel ``c = i // T_lat``, time ``t = i % T_lat``.
2. Each payload bit is written ``BIT_REPETITION`` times (default 5) at flat indices
   ``[i*R, i*R+1, …, i*R+R-1]`` (``R = BIT_REPETITION``).
3. Each scalar is clamped to ``[-1, 1]`` (Tanh bottleneck range), mapped to an integer
   ``q in [0, L-1]`` with ``L = quant_levels`` (default 16 — coarse step ~0.13).
4. Embed bit ``b``: force ``q' = (q & ~1) | b`` (LSB of ``q``).
5. Map ``q'`` back to float and write into ``z``.

Extraction reads the same indices from ``z_rec = encode(stego_audio)`` and applies
majority vote (``>= 3`` of 5) per bit.

Why coarse ``L``: AE round-trip drift on embedded coefficients (~0.03 std) is far larger
than the LSB step at ``L=2048`` (~0.0005), which drives ~50% BER. A coarse grid makes
the LSB decision margin robust; repetition absorbs residual per-index errors.

Limitations: lossy AE; capacity is ``floor(|z| / R)`` bits. ``--quant-levels`` can be
tuned (very low ``L`` improves BER, very high ``L`` degrades it).
"""

from __future__ import annotations

import csv
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
DEFAULT_QUANT_LEVELS = 16
BIT_REPETITION = 5
METHOD_LABEL = "Autoencoder Latent"

# Payload capacity sweep: usage as % of theoretical max latent bits (|z|/R).
CAPACITY_SWEEP_LEVELS_PCT = (25, 50, 75, 90, 100)

# Benchmark-audio suite: low-to-moderate latent usage per file.
BENCHMARK_USAGE_LEVELS_PCT = (1, 2, 5, 10, 15, 20, 25)
BENCHMARK_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac"}


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


def latent_capacity_bits(
    z: np.ndarray,
    *,
    repetition: int = BIT_REPETITION,
) -> int:
    """Payload bit capacity given ``repetition`` copies per bit."""
    return int(z.size) // max(1, repetition)


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


def _repetition_majority(votes: list[int], repetition: int) -> int:
    """Majority bit from ``repetition`` LSB reads (ties -> 0)."""
    need = repetition // 2 + 1
    return 1 if sum(votes) >= need else 0


def embed_bits_in_latent(
    z: np.ndarray,
    bits: str,
    *,
    quant_levels: int = DEFAULT_QUANT_LEVELS,
    start_index: int = 0,
    repetition: int = BIT_REPETITION,
) -> np.ndarray:
    """
    Copy ``z`` and embed ``bits`` with ``repetition`` LSB copies per bit.

    Bit ``i`` uses flat indices ``start_index + i*repetition + r`` for ``r in 0..R-1``.
    """
    out = np.array(z, dtype=np.float32, copy=True)
    flat = out.reshape(-1)
    rep = max(1, repetition)
    max_bits = (flat.size - start_index) // rep
    n = min(len(bits), max_bits)
    for i in range(n):
        b = int(bits[i])
        for r in range(rep):
            idx = start_index + i * rep + r
            flat[idx] = embed_bit_lsb(float(flat[idx]), b, quant_levels)
    return out


def extract_bits_from_latent(
    z: np.ndarray,
    n_bits: int,
    *,
    quant_levels: int = DEFAULT_QUANT_LEVELS,
    start_index: int = 0,
    repetition: int = BIT_REPETITION,
) -> str:
    """Read ``n_bits`` via majority vote over ``repetition`` LSB copies per bit."""
    flat = np.asarray(z, dtype=np.float32).reshape(-1)
    rep = max(1, repetition)
    max_bits = (flat.size - start_index) // rep
    n = min(n_bits, max_bits)
    out: list[str] = []
    for i in range(n):
        votes = [
            extract_bit_lsb(float(flat[start_index + i * rep + r]), quant_levels)
            for r in range(rep)
        ]
        out.append(str(_repetition_majority(votes, rep)))
    return "".join(out)


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


def print_embedding_plan(
    z: np.ndarray,
    payload: bytes,
    quant_levels: int,
    *,
    repetition: int = BIT_REPETITION,
) -> None:
    """Document which latent coefficients carry the secret."""
    t_lat = z.shape[-1]
    n_bits = len(payload) * 8
    rep = max(1, repetition)
    cap_bits = latent_capacity_bits(z, repetition=rep)
    last_idx = n_bits * rep - 1
    c0, t0 = index_to_channel_time(0, t_lat)
    c1, t1 = index_to_channel_time(min(last_idx, z.size - 1), t_lat)
    step = (LATENT_MAX - LATENT_MIN) / max(1, quant_levels - 1)
    print("\n--- Latent embedding plan ---")
    print(f"Model:              {MODEL_NAME}")
    print(f"Latent shape:       {tuple(z.shape)}  (C={z.shape[1]}, T_lat={t_lat})")
    print(f"Quant levels (L):   {quant_levels}  (LSB step ~{step:.4f})")
    print(f"Bit repetition:     {rep}x majority vote")
    print(f"Modified indices:   flat[0 : {n_bits * rep}]  ({n_bits} bits x {rep})")
    print(f"Index mapping:      (c, t) = (i // {t_lat}, i % {t_lat})")
    print(f"First coefficient:  index 0 -> channel {c0}, time {t0}")
    print(f"Last coefficient:   index {last_idx} -> channel {c1}, time {t1}")
    print(f"Payload bytes:      {len(payload)}  ({n_bits} bits)")
    print(
        f"Latent capacity:    {cap_bits} bits "
        f"({cap_bits // 8} bytes) with {rep}x repetition"
    )
    print(
        "Rule: clamp v in [-1,1] -> q=round(v); q=(q&~1)|bit; v'=dequantize(q'); "
        f"decode bit by majority over {rep} copies"
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


def stego_tensor_to_model_input(
    stego,
    *,
    num_samples: int = DEFAULT_NUM_SAMPLES,
) -> np.ndarray:
    """
    Decoder output -> float32 ``[1, 2, T]`` for re-encode (preserve L/R).

    Do not collapse to mono before extraction: ``encode(mono_dup)`` != ``encode(stego)``.
    """
    import torch

    if isinstance(stego, torch.Tensor):
        y = stego.detach().cpu().numpy()
    else:
        y = np.asarray(stego, dtype=np.float32)
    if y.ndim == 3:
        y = y[0]
    if y.ndim != 2 or y.shape[0] < 2:
        raise ValueError(f"Expected stereo [2, T], got shape {y.shape}")
    n = min(y.shape[1], num_samples)
    left = y[0, :n]
    right = y[1, :n]
    if n < num_samples:
        left = np.pad(left, (0, num_samples - n))
        right = np.pad(right, (0, num_samples - n))
    stereo = np.stack([left, right], axis=0)
    return stereo[np.newaxis, :, :].astype(np.float32)


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
    quiet: bool = False,
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
    if not quiet:
        print_embedding_plan(z_np, payload, quant_levels, repetition=BIT_REPETITION)
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
# Benchmark-audio discovery + suite
# ---------------------------------------------------------------------------


def _script_root() -> Path:
    return Path(__file__).resolve().parent


def default_benchmark_search_roots() -> list[Path]:
    """Candidate folders for benchmark audio (first existing wins per file)."""
    root = _script_root()
    repo = root.parent
    return [
        root / "audiosamples",
        repo / "audiosamples",
        root / "benchmark-audio",
        repo / "benchmark-audio",
        root / "audios",
        repo / "audios",
    ]


def discover_benchmark_audio_files(
    search_roots: list[Path] | None = None,
) -> list[Path]:
    """
    Recursively collect ``.wav``, ``.mp3``, ``.flac`` under benchmark folders.

    Searches ``audiosamples``, ``benchmark-audio``, and ``audios/`` (see
    ``default_benchmark_search_roots``). De-duplicates by resolved path.
    """
    roots = search_roots or default_benchmark_search_roots()
    found: dict[str, Path] = {}
    for base in roots:
        if not base.is_dir():
            continue
        for ext in BENCHMARK_AUDIO_EXTENSIONS:
            for path in base.rglob(f"*{ext}"):
                if path.is_file():
                    found[str(path.resolve())] = path.resolve()
    return sorted(found.values(), key=lambda p: p.name.lower())


def _infer_audio_category(path: Path) -> str:
    """Heuristic label from path/name: speech, music, transient, or general."""
    name = f"{path.parent.name}/{path.name}".lower()
    if any(k in name for k in ("speech", "voice", "vocal", "talk", "dialog")):
        return "speech"
    if any(k in name for k in ("drum", "perc", "click", "transient", "impact", "snare")):
        return "transient"
    if any(
        k in name
        for k in ("music", "song", "piano", "guitar", "orchestra", "melody", "chord")
    ):
        return "music"
    return "general"


def load_benchmark_audio_file(path: Path) -> tuple[np.ndarray, int, str]:
    """
    Load benchmark file as float64 ``[T, C]``.

    Returns ``(audio, sample_rate, channels_label)`` with channels_label
    ``mono`` or ``stereo``.
    """
    path = Path(path)
    ext = path.suffix.lower()
    try:
        data, sr = sf.read(path, dtype="float64", always_2d=True)
    except Exception:
        if ext == ".mp3":
            import librosa

            y, sr = librosa.load(str(path), sr=None, mono=False)
            if y.ndim == 1:
                data = y[:, np.newaxis]
            else:
                data = y.T
            data = np.asarray(data, dtype=np.float64)
        else:
            raise
    channels = "stereo" if data.shape[1] >= 2 else "mono"
    return data, int(sr), channels


def _payload_seed(audio_path: Path, usage_pct: int) -> int:
    return hash((str(audio_path.resolve()), usage_pct)) & 0xFFFFFFFF


def _run_benchmark_trial(
    ae,
    *,
    cover_in: np.ndarray,
    cover_np: np.ndarray,
    cover_samples: int,
    cover_size_bytes: int,
    max_payload_bits: int,
    usage_pct: int,
    audio_path: Path,
    quant_levels: int,
    num_samples: int,
    device: str,
) -> tuple[dict[str, float | str], np.ndarray]:
    """Single (audio, usage level) embed/extract/evaluate; returns row + stego mono."""
    from stego_analysis import (
        bit_error_rate_bytes,
        compute_audio_quality_metrics,
        time_embed_extract,
    )

    target_bits = max(1, int(np.floor(max_payload_bits * (usage_pct / 100.0))))
    payload_bytes_len = max(1, target_bits // 8)
    payload_bits = payload_bytes_len * 8
    rng = np.random.default_rng(_payload_seed(audio_path, usage_pct))
    payload = rng.integers(0, 256, size=payload_bytes_len, dtype=np.uint8).tobytes()
    state: dict[str, object] = {}

    def _embed() -> object:
        stego_t, _, _ = embed_latent_stego(
            ae,
            cover_in,
            payload,
            quant_levels=quant_levels,
            device=device,
            quiet=True,
        )
        state["stego_model_in"] = stego_tensor_to_model_input(
            stego_t, num_samples=num_samples
        )
        state["stego_np"] = model_output_to_numpy(stego_t)
        return stego_t

    def _extract() -> bytes:
        recovered, _ = extract_latent_payload(
            ae,
            np.asarray(state["stego_model_in"], dtype=np.float32),
            payload_bytes_len,
            quant_levels=quant_levels,
            device=device,
        )
        return recovered

    embed_sec, extract_sec, _, recovered = time_embed_extract(_embed, _extract)
    stego_np = np.asarray(state["stego_np"], dtype=np.float64)

    ber_pct = bit_error_rate_bytes(payload, recovered)
    bit_errors = int(round(ber_pct * payload_bits / 100.0)) if payload_bits else 0
    extracted_ok = int(recovered == payload)
    quality = compute_audio_quality_metrics(
        cover_np[:, np.newaxis], stego_np[:, np.newaxis], SAMPLE_RATE
    )
    payload_usage_pct = 100.0 * payload_bits / max_payload_bits
    payload_cover_ratio_pct = (
        100.0 * payload_bytes_len / cover_size_bytes if cover_size_bytes > 0 else 0.0
    )
    bps = payload_bits / cover_samples if cover_samples > 0 else 0.0

    row: dict[str, float | str] = {
        "filename": audio_path.name,
        "audio_path": str(audio_path),
        "audio_category": _infer_audio_category(audio_path),
        "duration_sec": float(len(cover_np) / SAMPLE_RATE),
        "sample_rate": float(SAMPLE_RATE),
        "channels": "mono",
        "cover_size_bytes": float(cover_size_bytes),
        "num_samples": float(cover_samples),
        "usage_target_pct": float(usage_pct),
        "usage_actual_pct": float(payload_usage_pct),
        "payload_size_bytes": float(payload_bytes_len),
        "payload_bits": float(payload_bits),
        "max_payload_bits": float(max_payload_bits),
        "payload_cover_ratio_pct": float(payload_cover_ratio_pct),
        "bits_per_sample": float(bps),
        "embed_sec": float(embed_sec),
        "extract_sec": float(extract_sec),
        "total_sec": float(embed_sec + extract_sec),
        "ber_pct": float(ber_pct),
        "bit_errors": float(bit_errors),
        "extracted_success": float(extracted_ok),
        "psnr_db": float(quality["psnr_db"]),
        "thd_stego_pct": float(quality["thd_stego_pct"]),
        "imd_stego_pct": float(quality["imd_stego_pct"]),
        "odg_approx": float(quality["odg_approx"]),
        "peaq_quality_approx": float(quality["peaq_quality_approx"]),
        "correlation": float(quality["correlation"]),
        "snr_db": float(quality["snr_db"]),
        "lsd_db": float(quality["lsd_db"]),
        "seaq_score": float(quality["seaq_score"]),
    }
    return row, stego_np


def run_benchmark_audio_suite(
    *,
    search_roots: list[Path] | None = None,
    quant_levels: int = DEFAULT_QUANT_LEVELS,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    device: str | None = None,
) -> list[dict[str, float | str]]:
    """
  Run latent stego on every benchmark audio × usage level (1–25%).

  Saves ``audio_out/analysis/autoencoder/benchmark_audio_results.csv`` and
  waveform/spectrogram plots for best and worst BER cases.
  """
    import os

    import torch

    os.environ.setdefault("MPLBACKEND", "Agg")
    from stego_analysis import format_bytes, run_baseline_visualization

    analysis_dir = _script_root() / "audio_out" / "analysis" / "autoencoder"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    csv_path = analysis_dir / "benchmark_audio_results.csv"
    plot_dir = analysis_dir / "benchmark_suite"
    plot_dir.mkdir(parents=True, exist_ok=True)

    audio_files = discover_benchmark_audio_files(search_roots)
    roots = search_roots or default_benchmark_search_roots()
    existing_roots = [r for r in roots if r.is_dir()]

    print("\n=== Autoencoder latent benchmark-audio suite ===")
    print(f"Quant levels (L): {quant_levels}, repetition: {BIT_REPETITION}x")
    print(f"Usage levels (% of max latent bits): {BENCHMARK_USAGE_LEVELS_PCT}")
    if existing_roots:
        print("Search roots:")
        for r in existing_roots:
            print(f"  - {r}")
    else:
        print("Search roots (none found on disk):")
        for r in roots:
            print(f"  - {r}")

    if not audio_files:
        print(
            "\nNo benchmark audio files found. Add .wav/.mp3/.flac under "
            "audiosamples/ or audios/ (recursive)."
        )
        return []

    print(f"Found {len(audio_files)} audio file(s).")

    ae, device = load_autoencoder(device)
    results: list[dict[str, float | str]] = []
    best_case: dict[str, object] | None = None
    worst_case: dict[str, object] | None = None

    for audio_path in audio_files:
        try:
            cover_audio, file_sr, channels = load_benchmark_audio_file(audio_path)
            file_size = int(audio_path.stat().st_size)
            duration_sec = float(len(cover_audio) / file_sr) if file_sr > 0 else 0.0

            cover_in, _ = prepare_cover_audio(
                cover_audio, file_sr, num_samples=num_samples, target_sr=SAMPLE_RATE
            )
            cover_np = 0.5 * (cover_in[0, 0] + cover_in[0, 1])
            cover_samples = int(len(cover_np))

            x = torch.from_numpy(cover_in).to(device)
            with torch.no_grad():
                z_cover = latent_to_numpy(_tensor_encode(ae, x))
            max_payload_bits = latent_capacity_bits(z_cover, repetition=BIT_REPETITION)

            print(
                f"\n--- {audio_path.name} ({channels}, {duration_sec:.2f}s, "
                f"{format_bytes(file_size)}) ---"
            )

            for usage_pct in BENCHMARK_USAGE_LEVELS_PCT:
                row, stego_np = _run_benchmark_trial(
                    ae,
                    cover_in=cover_in,
                    cover_np=cover_np,
                    cover_samples=cover_samples,
                    cover_size_bytes=file_size,
                    max_payload_bits=max_payload_bits,
                    usage_pct=usage_pct,
                    audio_path=audio_path,
                    quant_levels=quant_levels,
                    num_samples=num_samples,
                    device=device,
                )
                row["channels"] = channels
                row["duration_sec"] = duration_sec
                row["sample_rate"] = float(file_sr)
                results.append(row)

                ber = float(row["ber_pct"])
                case = {
                    "row": row,
                    "cover_np": cover_np.copy(),
                    "stego_np": stego_np.copy(),
                }
                if best_case is None or ber < float(best_case["row"]["ber_pct"]):
                    best_case = case
                if worst_case is None or ber > float(worst_case["row"]["ber_pct"]):
                    worst_case = case

        except Exception as exc:
            warnings.warn(f"Skipping {audio_path}: {exc}")

    if not results:
        print("\nNo successful benchmark runs.")
        return []

    csv_fields = [
        "filename",
        "audio_path",
        "audio_category",
        "duration_sec",
        "sample_rate",
        "channels",
        "cover_size_bytes",
        "num_samples",
        "usage_target_pct",
        "usage_actual_pct",
        "payload_size_bytes",
        "payload_bits",
        "max_payload_bits",
        "payload_cover_ratio_pct",
        "bits_per_sample",
        "embed_sec",
        "extract_sec",
        "total_sec",
        "ber_pct",
        "bit_errors",
        "extracted_success",
        "psnr_db",
        "thd_stego_pct",
        "imd_stego_pct",
        "odg_approx",
        "peaq_quality_approx",
        "correlation",
        "snr_db",
        "lsd_db",
        "seaq_score",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in csv_fields})

    print(f"\nSaved CSV: {csv_path}")

    # Per-file summary table
    headers = ["File", "Cat", "Usage", "BER", "PSNR", "Corr", "OK", "Runtime"]
    line = "|" + "|".join(f" {h:^8} " for h in headers) + "|"
    sep = "|" + "|".join("-" * 10 for _ in headers) + "|"
    print("\n--- Benchmark runs (all audio × usage) ---")
    print(line)
    print(sep)
    for r in results:
        ok = "yes" if int(float(r["extracted_success"])) else "no"
        print(
            "|"
            f" {str(r['filename'])[:8]:^8} "
            "|"
            f" {str(r['audio_category'])[:8]:^8} "
            "|"
            f" {float(r['usage_target_pct']):>6.0f}% "
            "|"
            f" {float(r['ber_pct']):>7.3f}% "
            "|"
            f" {float(r['psnr_db']):>6.1f} "
            "|"
            f" {float(r['correlation']):>6.4f} "
            "|"
            f" {ok:^8} "
            "|"
            f" {float(r['total_sec'])*1000:>6.0f}ms "
            "|"
        )

    # Averages per usage level
    print("\n--- Average metrics per payload usage level ---")
    avg_headers = ["Usage", "Avg BER", "Avg PSNR", "Avg Corr", "Runs"]
    avg_line = "|" + "|".join(f" {h:^10} " for h in avg_headers) + "|"
    avg_sep = "|" + "|".join("-" * 12 for _ in avg_headers) + "|"
    print(avg_line)
    print(avg_sep)
    for usage in BENCHMARK_USAGE_LEVELS_PCT:
        subset = [r for r in results if int(float(r["usage_target_pct"])) == usage]
        if not subset:
            continue
        avg_ber = float(np.mean([float(r["ber_pct"]) for r in subset]))
        avg_psnr = float(np.mean([float(r["psnr_db"]) for r in subset]))
        avg_corr = float(np.mean([float(r["correlation"]) for r in subset]))
        print(
            "|"
            f" {usage:>8}% "
            "|"
            f" {avg_ber:>9.4f}% "
            "|"
            f" {avg_psnr:>8.2f} dB "
            "|"
            f" {avg_corr:>10.6f} "
            "|"
            f" {len(subset):>10} "
            "|"
        )

    best_row = best_case["row"] if best_case else None
    worst_row = worst_case["row"] if worst_case else None
    if best_row is not None:
        print(
            f"\nBest BER:  {best_row['ber_pct']:.4f}% — "
            f"{best_row['filename']} @ {best_row['usage_target_pct']:.0f}% usage"
        )
    if worst_row is not None:
        print(
            f"Worst BER: {worst_row['ber_pct']:.4f}% — "
            f"{worst_row['filename']} @ {worst_row['usage_target_pct']:.0f}% usage"
        )

    # Category breakdown
    print("\n--- Average BER by inferred audio category ---")
    categories = sorted({str(r["audio_category"]) for r in results})
    cat_line = "|" + "|".join(f" {h:^12} " for h in ["Category", "Avg BER", "Avg PSNR", "N"]) + "|"
    cat_sep = "|" + "|".join("-" * 14 for _ in range(4)) + "|"
    print(cat_line)
    print(cat_sep)
    category_notes: list[str] = []
    for cat in categories:
        subset = [r for r in results if str(r["audio_category"]) == cat]
        avg_ber = float(np.mean([float(r["ber_pct"]) for r in subset]))
        avg_psnr = float(np.mean([float(r["psnr_db"]) for r in subset]))
        print(
            "|"
            f" {cat:^12} "
            "|"
            f" {avg_ber:>10.4f}% "
            "|"
            f" {avg_psnr:>10.2f} dB "
            "|"
            f" {len(subset):>12} "
            "|"
        )
        category_notes.append(f"{cat}: BER={avg_ber:.3f}%")

    if len(categories) > 1:
        bers = {
            c: float(np.mean([float(r["ber_pct"]) for r in results if r["audio_category"] == c]))
            for c in categories
        }
        spread = max(bers.values()) - min(bers.values())
        if spread > 5.0:
            print(
                "\nCategory behavior: inferred categories differ in recovery "
                f"(BER spread {spread:.2f} pp). Check path/folder naming for labels."
            )
        else:
            print(
                "\nCategory behavior: no strong difference between inferred "
                f"speech/music/transient labels (BER spread {spread:.2f} pp)."
            )
    else:
        print(
            "\nCategory behavior: only one category detected; rename or organize "
            "benchmark folders (speech/, music/, drum/, etc.) for comparisons."
        )

    # Plots for best / worst BER
    if best_case is not None:
        br = best_case["row"]
        tag = (
            f"best_ber_{Path(str(br['filename'])).stem}_"
            f"u{int(float(br['usage_target_pct']))}"
        )
        cover_m = np.asarray(best_case["cover_np"], dtype=np.float64)[:, np.newaxis]
        stego_m = np.asarray(best_case["stego_np"], dtype=np.float64)[:, np.newaxis]
        run_baseline_visualization(
            cover_m,
            stego_m,
            SAMPLE_RATE,
            tag,
            plot_dir,
        )
        plot_tag = tag.lower().replace(" ", "_")
        print(f"Saved best-BER plots: {plot_dir / f'{plot_tag}_waveform.png'}")

    if worst_case is not None:
        wr = worst_case["row"]
        tag = (
            f"worst_ber_{Path(str(wr['filename'])).stem}_"
            f"u{int(float(wr['usage_target_pct']))}"
        )
        cover_m = np.asarray(worst_case["cover_np"], dtype=np.float64)[:, np.newaxis]
        stego_m = np.asarray(worst_case["stego_np"], dtype=np.float64)[:, np.newaxis]
        run_baseline_visualization(
            cover_m,
            stego_m,
            SAMPLE_RATE,
            tag,
            plot_dir,
        )
        plot_tag = tag.lower().replace(" ", "_")
        print(f"Saved worst-BER plots: {plot_dir / f'{plot_tag}_waveform.png'}")

    print("\n--- Benchmark suite complete ---")
    return results


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
        state["stego_model_in"] = stego_tensor_to_model_input(
            stego_t, num_samples=num_samples
        )
        state["stego_np"] = model_output_to_numpy(stego_t)
        state["cover_np"] = 0.5 * (cover_in[0, 0] + cover_in[0, 1])
        return stego_t

    def _extract():
        recovered, _ = extract_latent_payload(
            ae,
            state["stego_model_in"],
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


def run_capacity_sweep(
    cover_audio: np.ndarray,
    sr: int,
    *,
    quant_levels: int = DEFAULT_QUANT_LEVELS,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    cover_path: Path | None = None,
    device: str | None = None,
) -> list[dict[str, float]]:
    """
    Sweep payload usage from 25% to 100% of theoretical latent capacity.

    Reuses the current PoC pipeline:
    cover -> encode -> embed -> decode -> re-encode -> recover -> evaluate.
    """
    import torch
    from stego_analysis import (
        bit_error_rate_bytes,
        compute_audio_quality_metrics,
        format_bytes,
        time_embed_extract,
    )

    analysis_dir = (
        Path(__file__).resolve().parent / "audio_out" / "analysis" / "autoencoder"
    )
    analysis_dir.mkdir(parents=True, exist_ok=True)
    sweep_audio_dir = Path(__file__).resolve().parent / "audio_out" / "capacity_sweep"
    sweep_audio_dir.mkdir(parents=True, exist_ok=True)
    csv_path = analysis_dir / "capacity_sweep_results.csv"

    ae, device = load_autoencoder(device)
    cover_in, _ = prepare_cover_audio(
        cover_audio, sr, num_samples=num_samples, target_sr=SAMPLE_RATE
    )
    cover_np = 0.5 * (cover_in[0, 0] + cover_in[0, 1])
    cover_samples = int(len(cover_np))

    # Use real on-disk WAV sizes for cover/stego growth reporting.
    cover_ref_path = sweep_audio_dir / "cover_reference.wav"
    sf.write(cover_ref_path, cover_np, SAMPLE_RATE)
    cover_size_bytes = (
        int(Path(cover_path).stat().st_size)
        if cover_path is not None and Path(cover_path).exists()
        else int(cover_ref_path.stat().st_size)
    )

    x = torch.from_numpy(cover_in).to(device)
    with torch.no_grad():
        z_cover = _tensor_encode(ae, x)
    z_cover_np = latent_to_numpy(z_cover)
    max_payload_bits = latent_capacity_bits(z_cover_np, repetition=BIT_REPETITION)

    results: list[dict[str, float]] = []
    rng = np.random.default_rng(20260526)

    print("\n=== Autoencoder latent capacity sweep ===")
    print(f"Latent shape: {tuple(z_cover_np.shape)}")
    print(
        f"Theoretical max payload bits: {max_payload_bits} "
        f"({max_payload_bits // 8} bytes) with {BIT_REPETITION}x repetition"
    )
    print(f"Quant levels (L): {quant_levels}")

    for usage_pct in CAPACITY_SWEEP_LEVELS_PCT:
        target_bits = int(np.floor(max_payload_bits * (usage_pct / 100.0)))
        payload_bytes_len = max(1, target_bits // 8)
        payload_bits = payload_bytes_len * 8
        payload = rng.integers(0, 256, size=payload_bytes_len, dtype=np.uint8).tobytes()
        stego_path = sweep_audio_dir / f"stego_usage_{usage_pct:03d}.wav"
        state: dict[str, object] = {}

        def _embed() -> object:
            stego_t, _, _ = embed_latent_stego(
                ae,
                cover_in,
                payload,
                quant_levels=quant_levels,
                device=device,
                quiet=True,
            )
            state["stego_t"] = stego_t
            state["stego_model_in"] = stego_tensor_to_model_input(
                stego_t, num_samples=num_samples
            )
            state["stego_np"] = model_output_to_numpy(stego_t)
            return stego_t

        def _extract() -> bytes:
            recovered, _ = extract_latent_payload(
                ae,
                np.asarray(state["stego_model_in"], dtype=np.float32),
                payload_bytes_len,
                quant_levels=quant_levels,
                device=device,
            )
            return recovered

        embed_sec, extract_sec, _, recovered = time_embed_extract(_embed, _extract)
        stego_np = np.asarray(state["stego_np"], dtype=np.float64)
        sf.write(stego_path, stego_np, SAMPLE_RATE)
        stego_size_bytes = int(stego_path.stat().st_size)
        size_growth_bytes = stego_size_bytes - cover_size_bytes
        size_growth_pct = (
            100.0 * size_growth_bytes / cover_size_bytes if cover_size_bytes > 0 else 0.0
        )

        ber_pct = bit_error_rate_bytes(payload, recovered)
        bit_errors = int(round(ber_pct * payload_bits / 100.0)) if payload_bits else 0
        extracted_ok = int(recovered == payload)
        quality = compute_audio_quality_metrics(
            cover_np[:, np.newaxis], stego_np[:, np.newaxis], SAMPLE_RATE
        )

        payload_usage_pct = 100.0 * payload_bits / max_payload_bits
        payload_cover_ratio_pct = (
            100.0 * payload_bytes_len / cover_size_bytes if cover_size_bytes > 0 else 0.0
        )
        bps = payload_bits / cover_samples if cover_samples > 0 else 0.0
        runtime_total_sec = embed_sec + extract_sec

        row = {
            "usage_target_pct": float(usage_pct),
            "usage_actual_pct": float(payload_usage_pct),
            "cover_samples": float(cover_samples),
            "cover_size_bytes": float(cover_size_bytes),
            "payload_size_bytes": float(payload_bytes_len),
            "payload_bits": float(payload_bits),
            "max_payload_bits": float(max_payload_bits),
            "payload_cover_ratio_pct": float(payload_cover_ratio_pct),
            "bits_per_sample": float(bps),
            "stego_size_bytes": float(stego_size_bytes),
            "stego_growth_bytes": float(size_growth_bytes),
            "stego_growth_pct": float(size_growth_pct),
            "embed_sec": float(embed_sec),
            "extract_sec": float(extract_sec),
            "total_sec": float(runtime_total_sec),
            "ber_pct": float(ber_pct),
            "bit_errors": float(bit_errors),
            "extracted_success": float(extracted_ok),
            "psnr_db": float(quality["psnr_db"]),
            "thd_stego_pct": float(quality["thd_stego_pct"]),
            "imd_stego_pct": float(quality["imd_stego_pct"]),
            "odg_approx": float(quality["odg_approx"]),
            "peaq_quality_approx": float(quality["peaq_quality_approx"]),
            "correlation": float(quality["correlation"]),
            "snr_db": float(quality["snr_db"]),
            "lsd_db": float(quality["lsd_db"]),
            "seaq_score": float(quality["seaq_score"]),
        }
        results.append(row)

        print(f"\n=== Capacity level: {usage_pct}% (actual {payload_usage_pct:.2f}%) ===")
        print("Capacity info:")
        print(f"  cover size:               {cover_size_bytes} B ({cover_size_bytes/1024:.2f} KB)")
        print(f"  number of audio samples:  {cover_samples}")
        print(f"  payload size:             {payload_bytes_len} B ({payload_bytes_len/1024:.2f} KB)")
        print(f"  payload bits:             {payload_bits}")
        print(f"  theoretical max bits:     {max_payload_bits}")
        print(f"  payload usage:            {payload_usage_pct:.2f}%")
        print(f"  payload:cover ratio:      {payload_cover_ratio_pct:.4f}%")
        print(f"  bits per sample (bps):    {bps:.6f}")

        print("File sizes:")
        print(f"  cover audio size:         {cover_size_bytes} B ({cover_size_bytes/1024:.2f} KB)")
        print(f"  stego audio size:         {stego_size_bytes} B ({stego_size_bytes/1024:.2f} KB)")
        print(f"  stego growth:             {size_growth_bytes:+d} B ({size_growth_pct:+.4f}%)")

        print("Runtime:")
        print(f"  embedding time:           {embed_sec*1000.0:.2f} ms")
        print(f"  extraction time:          {extract_sec*1000.0:.2f} ms")
        print(f"  total runtime:            {runtime_total_sec*1000.0:.2f} ms")

        print("Recovery:")
        print(f"  BER:                      {ber_pct:.4f}%")
        print(f"  bit errors:               {bit_errors} / {payload_bits}")
        print(f"  extracted payload success:{' yes' if extracted_ok else ' no'}")

        print("Audio quality:")
        print(f"  PSNR:                     {quality['psnr_db']:.4f} dB")
        print(f"  THD (stego):              {quality['thd_stego_pct']:.6f} %")
        print(f"  IMD (stego):              {quality['imd_stego_pct']:.6f} %")
        print(f"  ODG (approx):             {quality['odg_approx']:.4f}")
        print(f"  PEAQ quality (approx):    {quality['peaq_quality_approx']:.6f}")
        print(f"  correlation coefficient:  {quality['correlation']:.6f}")
        print(f"  SNR:                      {quality['snr_db']:.4f} dB")
        print(f"  LSD:                      {quality['lsd_db']:.4f} dB")
        print(f"  SEAQ:                     {quality['seaq_score']:.4f}")

    headers = [
        "Usage",
        "Payload",
        "Ratio",
        "BPS",
        "BER",
        "PSNR",
        "Corr",
        "Runtime",
    ]
    line = "|" + "|".join(f" {h:^10} " for h in headers) + "|"
    sep = "|" + "|".join("-" * 12 for _ in headers) + "|"
    print("\n--- Capacity sweep summary ---")
    print(line)
    print(sep)
    for r in results:
        print(
            "|"
            f" {r['usage_actual_pct']:>9.2f}% "
            "|"
            f" {format_bytes(int(r['payload_size_bytes'])):>10} "
            "|"
            f" {r['payload_cover_ratio_pct']:>9.3f}% "
            "|"
            f" {r['bits_per_sample']:>10.4f} "
            "|"
            f" {r['ber_pct']:>9.4f}% "
            "|"
            f" {r['psnr_db']:>8.2f}dB "
            "|"
            f" {r['correlation']:>10.6f} "
            "|"
            f" {r['total_sec']*1000.0:>8.1f}ms "
            "|"
        )

    csv_fields = [
        "usage_target_pct",
        "usage_actual_pct",
        "cover_size_bytes",
        "cover_samples",
        "payload_size_bytes",
        "payload_bits",
        "max_payload_bits",
        "payload_cover_ratio_pct",
        "bits_per_sample",
        "stego_size_bytes",
        "stego_growth_bytes",
        "stego_growth_pct",
        "embed_sec",
        "extract_sec",
        "total_sec",
        "ber_pct",
        "bit_errors",
        "extracted_success",
        "psnr_db",
        "thd_stego_pct",
        "imd_stego_pct",
        "odg_approx",
        "peaq_quality_approx",
        "correlation",
        "snr_db",
        "lsd_db",
        "seaq_score",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"\nSaved CSV: {csv_path}")

    ber_trend = ", ".join(
        f"{int(r['usage_target_pct'])}%: {r['ber_pct']:.3f}%" for r in results
    )
    first_ber_degrade = next((r for r in results if r["ber_pct"] > 0.0), None)
    degrade_label = (
        f"{int(first_ber_degrade['usage_target_pct'])}%"
        if first_ber_degrade is not None
        else "No BER degradation in tested range"
    )
    baseline_psnr = results[0]["psnr_db"]
    first_quality_degrade = next(
        (r for r in results if (baseline_psnr - r["psnr_db"]) > 2.0), None
    )
    quality_label = (
        f"{int(first_quality_degrade['usage_target_pct'])}%"
        if first_quality_degrade is not None
        else "No >2 dB PSNR drop in tested range"
    )
    if first_ber_degrade is None and first_quality_degrade is None:
        fail_first = "Neither recovery nor reconstruction failed noticeably."
    elif first_ber_degrade is None:
        fail_first = "Reconstruction quality degraded before recovery."
    elif first_quality_degrade is None:
        fail_first = "Recovery stayed stable before quality degradation."
    else:
        fail_first = (
            "Recovery failed first."
            if first_ber_degrade["usage_target_pct"]
            < first_quality_degrade["usage_target_pct"]
            else "Reconstruction quality degraded first."
        )

    print("\n--- Capacity sweep conclusions ---")
    print(f"1) BER trend across capacity: {ber_trend}")
    print(f"2) Payload level where BER begins degrading: {degrade_label}")
    print(f"3) Which fails first (recovery vs quality): {fail_first}")
    print(f"   Quality degradation threshold used: >2 dB PSNR drop (first at {quality_label}).")
    return results


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
        help=(
            "Latent quantization levels for LSB embedding (default 16; "
            "lower=more robust, higher=fragile after AE round-trip)."
        ),
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
    parser.add_argument(
        "--capacity-sweep",
        action="store_true",
        help="Run payload-capacity sweep (25/50/75/90/100% latent usage).",
    )
    parser.add_argument(
        "--benchmark-suite",
        action="store_true",
        help=(
            "Run benchmark-audio suite on audiosamples/ or audios/ "
            "(1/2/5/10/15/20/25%% latent usage per file)."
        ),
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=None,
        help="Optional root folder for benchmark audio (recursive .wav/.mp3/.flac).",
    )
    args = parser.parse_args()

    def _synthetic():
        t = np.arange(args.num_samples) / SAMPLE_RATE
        tone = 0.35 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.sin(2 * np.pi * 880 * t)
        return tone[:, np.newaxis], SAMPLE_RATE

    out_dir = Path(__file__).resolve().parent / "audio_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    stego_path = args.stego_out or (out_dir / "autoencoder_latent_stego.wav")

    try:
        if args.benchmark_suite and args.capacity_sweep:
            print("ERROR: Use only one of --benchmark-suite or --capacity-sweep.")
            raise SystemExit(2)
        if args.benchmark_suite:
            search_roots = [args.benchmark_dir] if args.benchmark_dir else None
            run_benchmark_audio_suite(
                search_roots=search_roots,
                quant_levels=args.quant_levels,
                num_samples=args.num_samples,
                device=args.device,
            )
        elif args.capacity_sweep:
            cover, sr, cover_path = load_cover_audio(
                args.cover, synthetic_builder=_synthetic
            )
            run_capacity_sweep(
                cover,
                sr,
                quant_levels=args.quant_levels,
                num_samples=args.num_samples,
                cover_path=cover_path,
                device=args.device,
            )
        else:
            cover, sr, cover_path = load_cover_audio(
                args.cover, synthetic_builder=_synthetic
            )
            pinfo = resolve_payload(
                payload_file=args.payload_file,
                payload_text=args.payload_text,
                default_text="Hidden in ArchiSound latent",
            )
            payload_bytes = payload_info_to_bytes(pinfo)
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
