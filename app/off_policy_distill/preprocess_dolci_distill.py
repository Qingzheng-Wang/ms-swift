"""
Preprocess Dolci stage5 dataset into two formats:

1. train_sft.jsonl — SFT format (messages + tools only, all samples)
2. train_opd.jsonl — Off-policy distillation format (voice samples with
   teacher_messages + audios; text samples have messages only)

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

        # Assistant with function calls → emit as tool_call role messages
        # ms-swift's _preprocess_function_call merges them into one assistant
        # message with <tool_call> tags via hermes agent_template
        if role == 'assistant' and fc:
            fc_list = fc if isinstance(fc, list) else json.loads(fc) if isinstance(fc, str) else [fc]
            for call in fc_list:
                call_str = call if isinstance(call, str) else json.dumps(call, ensure_ascii=False)
                student_messages.append({'role': 'tool_call', 'content': call_str})
                teacher_messages.append({'role': 'tool_call', 'content': call_str})
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

    # Skip samples where tool follows non-tool_call (missing tool-call)
    for j in range(len(student_messages)):
        if student_messages[j]['role'] == 'tool' and (j == 0 or student_messages[j - 1]['role'] not in ('tool_call', 'tool')):
            return None

    # Verify assistant/tool_call content is identical between student and teacher
    student_assistant = [m['content'] for m in student_messages if m['role'] in ('assistant', 'tool_call')]
    teacher_assistant = [m['content'] for m in teacher_messages if m['role'] in ('assistant', 'tool_call')]
    if student_assistant != teacher_assistant:
        print(f'WARNING: assistant content mismatch for {sample_id}, skipping')
        return None

    tools_str = json.dumps(tools, ensure_ascii=False) if tools else ''
    is_voice = data_type == 'voice' and bool(audios) # audios be non-empty

    # SFT: messages + tools + audios
    sft = {
        'messages': student_messages,
        'audios': audios if is_voice else [''],
        'tools': tools_str,
    }

    # OPD: + teacher_messages
    opd = {
        'messages': student_messages,
        'teacher_messages': teacher_messages,
        'audios': audios if is_voice else [''],
        'tools': tools_str,
    }

    return sft, opd


def process_chunk(args: tuple) -> tuple:
    """Process a chunk of lines. Returns (sft_path, opd_path, kept, skipped)."""
    chunk_idx, lines = args
    tmp_dir = OUTPUT_DIR / 'tmp'
    tmp_dir.mkdir(parents=True, exist_ok=True)
    sft_path = str(tmp_dir / f'sft_chunk_{chunk_idx}.jsonl')
    opd_path = str(tmp_dir / f'opd_chunk_{chunk_idx}.jsonl')

    kept = 0
    skipped = 0
    with open(sft_path, 'w', encoding='utf-8') as f_sft, \
         open(opd_path, 'w', encoding='utf-8') as f_opd:
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            result = convert_sample(raw)
            if result is None:
                skipped += 1
                continue

            sft, opd = result
            f_sft.write(json.dumps(sft, ensure_ascii=False) + '\n')
            f_opd.write(json.dumps(opd, ensure_ascii=False) + '\n')
            kept += 1

    return sft_path, opd_path, kept, skipped


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
    sft_output = output_dir / 'train_sft.jsonl'
    opd_output = output_dir / 'train_opd.jsonl'

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
    sft_paths = []
    opd_paths = []

    with Pool(processes=args.workers) as pool:
        for result in pool.imap(process_chunk, chunks):
            sft_path, opd_path, kept, skipped = result
            sft_paths.append(sft_path)
            opd_paths.append(opd_path)
            total_kept += kept
            total_skipped += skipped
            print(f'  Chunk done: +{kept} kept, +{skipped} skipped '
                  f'(running total: {total_kept}/{total_kept + total_skipped})')

    # Merge SFT chunks
    print(f'Merging SFT chunks into {sft_output} ...')
    with open(sft_output, 'w', encoding='utf-8') as fout:
        for tmp_path in sft_paths:
            with open(tmp_path, 'r', encoding='utf-8') as fin:
                for line in fin:
                    fout.write(line)
            os.remove(tmp_path)

    # Merge OPD chunks
    print(f'Merging OPD chunks into {opd_output} ...')
    with open(opd_output, 'w', encoding='utf-8') as fout:
        for tmp_path in opd_paths:
            with open(tmp_path, 'r', encoding='utf-8') as fin:
                for line in fin:
                    fout.write(line)
            os.remove(tmp_path)

    # Cleanup tmp dir
    tmp_dir = output_dir / 'tmp'
    if tmp_dir.exists():
        try:
            tmp_dir.rmdir()
        except OSError:
            pass

    print(f'\nDone!')
    print(f'  Input:   {total}')
    print(f'  Kept:    {total_kept} ({total_kept / total * 100:.1f}%)')
    print(f'  Skipped: {total_skipped} ({total_skipped / total * 100:.1f}%)')
    print(f'  SFT:     {sft_output}')
    print(f'  OPD:     {opd_output}')


if __name__ == '__main__':
    main()
