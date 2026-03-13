"""
Preprocess Dolci-Instruct-SFT dataset for ms-swift.

Handles three cases:
1. Pure text conversations → pass through
2. Tool call (assistant content=None + environment role) → convert to ms-swift tool format
3. Audio (user turn has corresponding .opus file) → add <audio> tag + audios field

Usage:
    python preprocess_dolci.py --workers 16
    python preprocess_dolci.py --workers 16 --split test
"""

import argparse
import json
import os
import tempfile
from multiprocessing import Pool
from pathlib import Path

DATA_ROOT = Path("/mnt/nas/lengyue/flash-fish-data/allenai/Dolci-Instruct-SFT")
AUDIO_DIR = DATA_ROOT / "audio"
OUTPUT_DIR = DATA_ROOT / "swift_format"


def convert_sample(raw: dict) -> dict | None:
    """Convert a single Dolci sample to ms-swift format.

    Returns None if the sample should be skipped (no valid assistant response).
    """
    sample_id = raw.get("id", "")
    messages = raw.get("messages", [])
    if not messages:
        return None

    # Check if audio dir exists for this sample
    audio_id = sample_id.replace("/", "_")
    audio_sample_dir = AUDIO_DIR / audio_id
    has_audio_dir = audio_sample_dir.is_dir()

    new_messages = []
    audios = []
    has_valid_assistant = False

    for msg_idx, msg in enumerate(messages):
        role = msg["role"]
        content = msg.get("content")

        # Case 2a: assistant with content=None (tool call initiation)
        if role == "assistant" and content is None:
            fc = msg.get("function_calls")
            if fc:
                new_messages.append({
                    "role": "assistant",
                    "content": json.dumps(fc, ensure_ascii=False),
                })
            else:
                # No function_calls and no content → truncated sample, skip this turn
                new_messages.append({"role": "assistant", "content": ""})
            continue

        # Case 2b: environment → tool
        if role == "environment":
            new_messages.append({"role": "tool", "content": content or ""})
            continue

        # Skip unknown roles
        if role not in ("system", "user", "assistant"):
            continue

        if role == "assistant" and content:
            has_valid_assistant = True

        # Case 3: user message with possible audio
        if role == "user" and has_audio_dir:
            audio_path = audio_sample_dir / f"{msg_idx:04d}.opus"
            if audio_path.exists():
                audios.append(str(audio_path))
                new_messages.append({
                    "role": "user",
                    "content": f"<audio>{content}" if content else "<audio>",
                })
                continue

        # Case 1: normal message
        new_messages.append({"role": role, "content": content})

    if not has_valid_assistant:
        return None

    result = {"messages": new_messages}
    if audios:
        result["audios"] = audios
    return result


def process_chunk(args: tuple) -> tuple:
    """Process a chunk of lines. Returns (output_path, kept_count, skipped_count)."""
    chunk_idx, lines = args
    tmp_path = os.path.join(tempfile.gettempdir(), f"dolci_chunk_{chunk_idx}.jsonl")

    kept = 0
    skipped = 0
    with open(tmp_path, "w", encoding="utf-8") as fout:
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            converted = convert_sample(raw)
            if converted is None:
                skipped += 1
                continue

            fout.write(json.dumps(converted, ensure_ascii=False) + "\n")
            kept += 1

    return tmp_path, kept, skipped


def main():
    parser = argparse.ArgumentParser(description="Preprocess Dolci-Instruct-SFT for ms-swift")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=50000,
                        help="Number of lines per chunk for parallel processing")
    args = parser.parse_args()

    input_path = DATA_ROOT / f"{args.split}.jsonl"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{args.split}.jsonl"

    print(f"Reading {input_path} ...")
    with open(input_path, "r", encoding="utf-8") as f:
        all_lines = f.readlines()
    total = len(all_lines)
    print(f"Total lines: {total}")

    # Split into chunks
    chunks = []
    for i in range(0, total, args.chunk_size):
        chunks.append((len(chunks), all_lines[i:i + args.chunk_size]))
    print(f"Split into {len(chunks)} chunks, processing with {args.workers} workers ...")

    # Free memory - lines are now in chunks
    del all_lines

    # Parallel processing
    total_kept = 0
    total_skipped = 0
    tmp_paths = []

    with Pool(processes=args.workers) as pool:
        for result in pool.imap(process_chunk, chunks):
            tmp_path, kept, skipped = result
            tmp_paths.append(tmp_path)
            total_kept += kept
            total_skipped += skipped
            print(f"  Chunk done: +{kept} kept, +{skipped} skipped "
                  f"(running total: {total_kept}/{total_kept + total_skipped})")

    # Merge chunks
    print(f"Merging {len(tmp_paths)} chunks into {output_path} ...")
    with open(output_path, "w", encoding="utf-8") as fout:
        for tmp_path in tmp_paths:
            with open(tmp_path, "r", encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)
            os.remove(tmp_path)

    print(f"\nDone!")
    print(f"  Input:   {total}")
    print(f"  Kept:    {total_kept} ({total_kept / total * 100:.1f}%)")
    print(f"  Skipped: {total_skipped} ({total_skipped / total * 100:.1f}%)")
    print(f"  Output:  {output_path}")


if __name__ == "__main__":
    main()
