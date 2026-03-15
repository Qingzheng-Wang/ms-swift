"""
Preprocess Dolci stage5 dataset for off-policy distillation (GKD).

Produces dual-format data:
- messages: student view (audio samples use <audio> placeholder)
- teacher_messages: teacher view (text-only, uses rewritten_query)

For text-only samples, teacher_messages is omitted (GKD trainer falls back to standard flow).

Usage:
    python preprocess_dolci_distill.py
    python preprocess_dolci_distill.py --workers 16
"""

import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

INPUT_PATH = Path('/home/qingzhengw/voice-agent-data-engine/output/stage5/dolci_stage5.jsonl')
INPUT_ROOT = Path('/home/qingzhengw/voice-agent-data-engine/output')
OUTPUT_DIR = Path('data') / 'dolci_qingzheng_swift_format'


def convert_sample(raw: dict) -> dict | None:
    """Convert a single Dolci stage5 sample to ms-swift GKD format.

    Returns None if the sample should be skipped.
    """
    sample_id = raw.get('id', '')
    turns = raw.get('turns', [])
    data_type = raw.get('data_type', 'text')
    function_schemas = raw.get('function_schemas', [])

    if not turns:
        return None

    student_messages = []
    teacher_messages = []
    audios = []
    tools = None
    has_valid_assistant = False

    # Extract function schemas as tools
    if function_schemas:
        tools = function_schemas

    for turn in turns:
        role = turn.get('role', '')
        content = turn.get('content', '')
        audio_path = turn.get('audio_path')
        rewritten_query = turn.get('rewritten_query')
        fc = turn.get('function_calls', [])
        tool_response = turn.get('tool_response')

        # Skip unknown roles
        if role not in ('user', 'assistant', 'system', 'tool'):
            continue

        # Tool response: content is null, actual data is in tool_response field
        if role == 'tool':
            tool_content = tool_response or content or ''
            if not isinstance(tool_content, str):
                tool_content = json.dumps(tool_content, ensure_ascii=False)
            student_messages.append({'role': 'tool', 'content': tool_content})
            teacher_messages.append({'role': 'tool', 'content': tool_content})
            continue

        # Assistant with function calls (tool call initiation)
        if role == 'assistant' and fc:
            fc_content = fc if isinstance(fc, str) else json.dumps(fc, ensure_ascii=False)
            student_messages.append({'role': 'assistant', 'content': fc_content})
            teacher_messages.append({'role': 'assistant', 'content': fc_content})
            has_valid_assistant = True
            continue

        if role == 'assistant' and content:
            has_valid_assistant = True

        # User message: handle audio vs text
        if role == 'user' and data_type == 'voice' and audio_path:
            # Student sees <audio>
            student_messages.append({'role': 'user', 'content': '<audio>'})
            audio_path = str(INPUT_ROOT / audio_path)
            audios.append(audio_path)

            # Teacher sees text version: prefer original content
            teacher_text = content or rewritten_query or ''
            teacher_messages.append({'role': 'user', 'content': teacher_text})
            continue

        # Normal message (same for both)
        student_messages.append({'role': role, 'content': content})
        teacher_messages.append({'role': role, 'content': content})

    if not has_valid_assistant:
        return None

    # Merge consecutive tool responses into one
    for msg_list in (student_messages, teacher_messages):
        merged = []
        for m in msg_list:
            if m['role'] == 'tool' and merged and merged[-1]['role'] == 'tool':
                merged[-1]['content'] += '\n' + m['content']
            else:
                merged.append(m)
        msg_list.clear()
        msg_list.extend(merged)

    # Skip samples where tool follows non-assistant (missing tool-call)
    for j in range(len(student_messages)):
        if student_messages[j]['role'] == 'tool' and (j == 0 or student_messages[j - 1]['role'] != 'assistant'):
            return None

    # Verify assistant content is identical between student and teacher
    student_assistant = [m['content'] for m in student_messages if m['role'] == 'assistant']
    teacher_assistant = [m['content'] for m in teacher_messages if m['role'] == 'assistant']
    if student_assistant != teacher_assistant:
        print(f'WARNING: assistant content mismatch for {sample_id}, skipping')
        return None

    result = {'messages': student_messages}

    # Only add teacher_messages for voice samples (text samples use same input for both)
    if data_type == 'voice' and audios:
        result['teacher_messages'] = teacher_messages
        result['audios'] = audios

    if tools:
        result['tools'] = json.dumps(tools, ensure_ascii=False)

    return result


def process_chunk(args: tuple) -> tuple:
    """Process a chunk of lines. Returns (output_path, kept_count, skipped_count)."""
    chunk_idx, lines = args
    tmp_dir = OUTPUT_DIR / 'tmp'
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = str(tmp_dir / f'dolci_gkd_chunk_{chunk_idx}.jsonl')

    kept = 0
    skipped = 0
    with open(tmp_path, 'w', encoding='utf-8') as fout:
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

            fout.write(json.dumps(converted, ensure_ascii=False) + '\n')
            kept += 1

    return tmp_path, kept, skipped


def main():
    num_cpus = os.cpu_count() or 1
    print(f'Detected {num_cpus} CPU cores. Defaulting to --workers {num_cpus}.')
    parser = argparse.ArgumentParser(description='Preprocess Dolci stage5 for GKD distillation')
    parser.add_argument('--input', type=str, default=str(INPUT_PATH), help='Input JSONL path')
    parser.add_argument('--output-dir', type=str, default=str(OUTPUT_DIR), help='Output directory')
    parser.add_argument('--workers', type=int, default=num_cpus)
    parser.add_argument('--chunk-size', type=int, default=20000,
                        help='Number of lines per chunk for parallel processing')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'train.jsonl'

    print(f'Reading {input_path} ...')
    with open(input_path, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
    total = len(all_lines)
    print(f'Total lines: {total}')

    # Split into chunks
    chunks = []
    for i in range(0, total, args.chunk_size):
        chunks.append((len(chunks), all_lines[i:i + args.chunk_size]))
    print(f'Split into {len(chunks)} chunks, processing with {args.workers} workers ...')

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
            print(f'  Chunk done: +{kept} kept, +{skipped} skipped '
                  f'(running total: {total_kept}/{total_kept + total_skipped})')

    # Merge chunks
    print(f'Merging {len(tmp_paths)} chunks into {output_path} ...')
    with open(output_path, 'w', encoding='utf-8') as fout:
        for tmp_path in tmp_paths:
            with open(tmp_path, 'r', encoding='utf-8') as fin:
                for line in fin:
                    fout.write(line)
            os.remove(tmp_path)

    # Count voice vs text samples
    voice_count = 0
    text_count = 0
    with open(output_path, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line)
            if 'teacher_messages' in sample:
                voice_count += 1
            else:
                text_count += 1

    print(f'\nDone!')
    print(f'  Input:   {total}')
    print(f'  Kept:    {total_kept} ({total_kept / total * 100:.1f}%)')
    print(f'  Skipped: {total_skipped} ({total_skipped / total * 100:.1f}%)')
    print(f'  Voice:   {voice_count}')
    print(f'  Text:    {text_count}')
    print(f'  Output:  {output_path}')


if __name__ == '__main__':
    main()
