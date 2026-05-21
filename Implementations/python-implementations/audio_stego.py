"""
Python port of AudioStego (main.cpp + Algorithm.cpp).

Embeds data by overwriting whole bytes in the PCM region (spread sampling),
not by flipping individual sample LSBs. Uses soundfile for WAV I/O and numpy
for buffer views.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

# Mirrors Algorithm.h
SUCCESS = 1
ERROR = 2
DEF_MODULE = 64
WAV_HEADER = 44
MY_HEADER_MODULE = 64
START_SPACE = 0
MY_HEADER = 9
END_MARKER = "@<;;"


def get_file_extension(path: str | Path) -> str:
    name = str(path)
    dot = name.rfind(".")
    return name[dot + 1 :] if dot != -1 else ""


def create_header(modulus: int, file_extension: str, is_binary: bool) -> bytes:
    """9-byte header: 4-byte modulus (int32 LE), 4-byte extension, 1-byte type."""
    if modulus <= 42946729:
        mod_bytes = struct.pack("<i", modulus)
    else:
        mod_bytes = bytes([DEF_MODULE, 0, 0, 0])

    ext = (file_extension[:4]).ljust(4)
    msg_type = b"b" if is_binary else b"t"
    return mod_bytes + ext.encode("ascii", errors="replace")[:4].ljust(4) + msg_type


def parse_header(header: bytes) -> tuple[int, str, str]:
    modulus = struct.unpack("<i", header[:4])[0]
    extension = header[4:8].decode("ascii", errors="replace").strip()
    msg_type = chr(header[8])
    return modulus, extension, msg_type


def load_wav_pcm_bytes(path: Path) -> tuple[bytearray, int, str]:
    """Read WAV via soundfile; return mutable PCM bytes, sample rate, subtype."""
    data, sr = sf.read(path, dtype="int16", always_2d=False)
    info = sf.info(path)
    pcm = bytearray(np.asarray(data, dtype=np.int16).tobytes())
    return pcm, sr, info.subtype


def save_wav_from_pcm(path: Path, pcm: bytearray, sr: int, subtype: str = "PCM_16") -> None:
    samples = np.frombuffer(pcm, dtype=np.int16)
    sf.write(path, samples, sr, subtype=subtype)


def load_file_buffer(path: Path) -> bytearray:
    """Full-file binary load (matches C++ ifstream). Works for .wav and .mp3."""
    return bytearray(path.read_bytes())


def save_file_buffer(buffer: bytearray, extension: str) -> Path:
    name = "output" if not extension else f"output.{extension}"
    out = Path(name)
    out.write_bytes(buffer)
    return out


def spreading_modulus(pcm_size: int, payload_size: int) -> int:
    return (pcm_size - START_SPACE) // (payload_size + MY_HEADER)


def write_spread_header(pcm: bytearray, custom_header: bytes) -> int:
    """Write 9 header bytes every MY_HEADER_MODULE bytes; return last index n."""
    n = 0
    pos = 0
    start = START_SPACE
    i = start
    while i < len(pcm):
        if n % MY_HEADER_MODULE == 0:
            pcm[i] = custom_header[pos]
            pos += 1
            if pos == MY_HEADER:
                break
        n += 1
        i += 1
    return n


def write_spread_payload(
    pcm: bytearray, payload: bytes, modulus: int, header_end_n: int
) -> int:
    """Embed payload every `modulus` bytes after header region."""
    j = 0
    pos = 0
    start = START_SPACE + header_end_n + MY_HEADER_MODULE
    i = start
    while i < len(pcm):
        if j % modulus == 0:
            pcm[i] = payload[pos]
            pos += 1
            if pos >= len(payload):
                break
        j += 1
        i += 1
    return pos


def read_spread_header(pcm: bytearray) -> tuple[bytes, int]:
    n = 0
    pos = 0
    custom = bytearray(MY_HEADER)
    i = START_SPACE
    while i < len(pcm):
        if n % MY_HEADER_MODULE == 0:
            custom[pos] = pcm[i]
            pos += 1
            if pos == MY_HEADER:
                break
        n += 1
        i += 1
    return bytes(custom), n


def read_spread_until_marker(
    pcm: bytearray, modulus: int, last_pos: int
) -> tuple[bytearray, bool]:
    """Read bytes every `modulus`; stop when '@<;;' is seen (same as C++)."""
    n = 0
    collected = bytearray()
    start = START_SPACE + last_pos + MY_HEADER_MODULE
    i = start

    def peek_ahead(extra: int) -> int:
        # C++: buffer.begin() + n + WAV_HEADER + MY_HEADER_MODULE + START_SPACE + lastPos + extra
        idx = start + n + extra
        return pcm[idx] if 0 <= idx < len(pcm) else -1

    while i < len(pcm):
        if n % modulus == 0:
            if pcm[i] == ord("@"):
                if (
                    peek_ahead(modulus) == ord("<")
                    and peek_ahead(2 * modulus) == ord(";")
                    and peek_ahead(3 * modulus) == ord(";")
                ):
                    return collected, True
            collected.append(pcm[i])
        n += 1
        i += 1
    return collected, False


def hide_in_wav_pcm(
    pcm: bytearray,
    payload: bytes,
    file_extension: str,
    is_binary: bool,
) -> int:
    modulus = spreading_modulus(len(pcm), len(payload))
    print(f"Spreading level: {modulus}")

    if modulus <= 3:
        print("The message might be to big for the audio file")
        return ERROR

    header = create_header(modulus, file_extension, is_binary)
    last_n = write_spread_header(pcm, header)
    print("Header wrote")

    written = write_spread_payload(pcm, payload, modulus, last_n)
    if written < len(payload):
        print("Maybe the whole file was not written in")

    return SUCCESS


def hide_string_wav(input_path: Path, message: str, input_ext: str) -> int:
    pcm, sr, subtype = load_wav_pcm_bytes(input_path)
    payload = (message + END_MARKER).encode("latin-1", errors="replace")
    status = hide_in_wav_pcm(pcm, payload, "", is_binary=False)
    if status != SUCCESS:
        return status
    out = Path("output.wav") if not input_ext else Path(f"output.{input_ext}")
    save_wav_from_pcm(out, pcm, sr, subtype)
    print(f"File has been saved as: {out}")
    return SUCCESS


def hide_binary_wav(
    input_path: Path, file_to_hide: Path, hide_ext: str, input_ext: str
) -> int:
    pcm, sr, subtype = load_wav_pcm_bytes(input_path)
    payload = file_to_hide.read_bytes() + END_MARKER.encode("ascii")
    status = hide_in_wav_pcm(pcm, payload, hide_ext, is_binary=True)
    if status != SUCCESS:
        return status
    out = Path("output.wav") if not input_ext else Path(f"output.{input_ext}")
    save_wav_from_pcm(out, pcm, sr, subtype)
    print(f"File has been saved as: {out}")
    return SUCCESS


def hide_string_raw(buffer: bytearray, message: str, input_ext: str) -> int:
    """C++-compatible: embed in bytes after WAV_HEADER in full file buffer."""
    payload = (message + END_MARKER).encode("latin-1", errors="replace")
    region = buffer[WAV_HEADER:]
    status = hide_in_wav_pcm(region, payload, "", is_binary=False)
    if status != SUCCESS:
        return status
    out = save_file_buffer(buffer, input_ext)
    print(f"File has been saved as: {out}")
    return SUCCESS


def find_hidden_wav_pcm(pcm: bytearray) -> int:
    print("Looking for the hidden message...")
    header_bytes, last_pos = read_spread_header(pcm)
    modulus, extension, msg_type = parse_header(header_bytes)

    if msg_type == "b":
        print("File detected. Retrieving it...")
        data, ok = read_spread_until_marker(pcm, modulus, last_pos)
        if not ok:
            print("Could not find the end tags of the hidden file :(")
            return ERROR
        print(f"Message recovered size: {len(data)} bytes")
        out = Path("output") if not extension else Path(f"output.{extension}")
        out.write_bytes(data)
        print(f"File has been saved as: {out}")
        return SUCCESS

    if msg_type == "t":
        print("String detected. Retrieving it...")
        data, ok = read_spread_until_marker(pcm, modulus, last_pos)
        if not ok:
            print("No message found :(")
            return ERROR
        print(f"Message recovered size: {len(data)} bytes")
        print(f"Message: {data.decode('latin-1', errors='replace')}")
        return SUCCESS

    print("Failed to detect a hidden file.")
    print("No custom header was found.")
    return ERROR


def find_hidden_raw(buffer: bytearray) -> int:
    return find_hidden_wav_pcm(buffer[WAV_HEADER:])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audio steganography (Python port)")
    parser.add_argument("input_file", nargs="?", help="Cover or stego audio/file")
    parser.add_argument(
        "message_or_flag",
        nargs="?",
        help="Text in 'quotes', file to hide, or -f/--find",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.input_file:
        print("Where are my the parameters mate?")
        print('Hide string:  python audio_stego.py input.wav "\'message\'"')
        print("Hide file:    python audio_stego.py input.wav secret.bin")
        print("Extract:      python audio_stego.py stego.wav -f")
        return 0

    input_path = Path(args.input_file)
    mode = 1  # hide string

    if args.message_or_flag is None:
        print("No message to hide was specified. Using a default string message...")
        message = "Default Message KEKLIFE"
        input_ext = get_file_extension(input_path)
        print("Doing it boss!")
        if input_path.suffix.lower() == ".wav":
            return hide_string_wav(input_path, message, input_ext)
        buffer = load_file_buffer(input_path)
        return hide_string_raw(buffer, message, input_ext)

    if args.message_or_flag in ("-f", "--find"):
        mode = 2
        print("Doing it boss!")
        if input_path.suffix.lower() == ".wav":
            pcm, _, _ = load_wav_pcm_bytes(input_path)
            status = find_hidden_wav_pcm(pcm)
        else:
            status = find_hidden_raw(load_file_buffer(input_path))
        print(
            "Recovering process has finished successfully.\nCleaning memory..."
            if status == SUCCESS
            else "Something failed.\nCleaning memory..."
        )
        return 0

    if len(args.message_or_flag) >= 2 and args.message_or_flag[0] == "'" and args.message_or_flag[-1] == "'":
        message = args.message_or_flag
        input_ext = get_file_extension(input_path)
        print("Doing it boss!")
        if input_path.suffix.lower() == ".wav":
            return hide_string_wav(input_path, message, input_ext)
        buffer = load_file_buffer(input_path)
        return hide_string_raw(buffer, message, input_ext)

    # Hide binary file
    hide_path = Path(args.message_or_flag)
    file_ext = get_file_extension(hide_path)
    input_ext = get_file_extension(input_path)
    print("Doing it boss!")
    if input_path.suffix.lower() == ".wav":
        return hide_binary_wav(input_path, hide_path, file_ext, input_ext)
    buffer = load_file_buffer(input_path)
    payload = hide_path.read_bytes() + END_MARKER.encode("ascii")
    region = buffer[WAV_HEADER:]
    status = hide_in_wav_pcm(region, payload, file_ext, is_binary=True)
    if status == SUCCESS:
        save_file_buffer(buffer, input_ext)
    return status


if __name__ == "__main__":
    sys.exit(main() or 0)
