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
INPUT_ROOT = Path('/home/qingzhengw/voice-agent-data-engine')
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
        'audios': audios if is_voice else [],
        'tools': tools_str,
    }

    # OPD: + teacher_messages
    opd = {
        'messages': student_messages,
        'teacher_messages': teacher_messages,
        'audios': audios if is_voice else [],
        'tools': tools_str,
    }

    return sft, opd


def process_chunk(args: tuple) -> tuple:
    """Process a chunk. Voice and text samples go to separate tmp files
    so that voice samples (with non-empty audios) can be written first
    during merge, ensuring HF datasets infers list<string> for audios."""
    chunk_idx, lines = args
    tmp_dir = OUTPUT_DIR / 'tmp'
    tmp_dir.mkdir(parents=True, exist_ok=True)
    sft_voice_path = str(tmp_dir / f'sft_voice_{chunk_idx}.jsonl')
    sft_text_path = str(tmp_dir / f'sft_text_{chunk_idx}.jsonl')
    opd_voice_path = str(tmp_dir / f'opd_voice_{chunk_idx}.jsonl')
    opd_text_path = str(tmp_dir / f'opd_text_{chunk_idx}.jsonl')

    voice_count = 0
    text_count = 0
    skipped = 0
    with open(sft_voice_path, 'w', encoding='utf-8') as f_sft_v, \
         open(sft_text_path, 'w', encoding='utf-8') as f_sft_t, \
         open(opd_voice_path, 'w', encoding='utf-8') as f_opd_v, \
         open(opd_text_path, 'w', encoding='utf-8') as f_opd_t:
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
            if sft['audios']:  # voice sample
                f_sft_v.write(json.dumps(sft, ensure_ascii=False) + '\n')
                f_opd_v.write(json.dumps(opd, ensure_ascii=False) + '\n')
                voice_count += 1
            else:
                f_sft_t.write(json.dumps(sft, ensure_ascii=False) + '\n')
                f_opd_t.write(json.dumps(opd, ensure_ascii=False) + '\n')
                text_count += 1

    return (sft_voice_path, sft_text_path, opd_voice_path, opd_text_path,
            voice_count, text_count, skipped)


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
    total_voice = 0
    total_text = 0
    total_skipped = 0
    sft_voice_paths = []
    sft_text_paths = []
    opd_voice_paths = []
    opd_text_paths = []

    with Pool(processes=args.workers) as pool:
        for result in pool.imap(process_chunk, chunks):
            sft_vp, sft_tp, opd_vp, opd_tp, vc, tc, skipped = result
            sft_voice_paths.append(sft_vp)
            sft_text_paths.append(sft_tp)
            opd_voice_paths.append(opd_vp)
            opd_text_paths.append(opd_tp)
            total_voice += vc
            total_text += tc
            total_skipped += skipped
            print(f'  Chunk done: voice +{vc}, text +{tc}, skipped +{skipped} '
                  f'(running: voice={total_voice}, text={total_text})')

    def merge_files(out_path, path_lists):
        """Merge tmp files into output. path_lists is a list of lists,
        written in order (voice first, then text)."""
        with open(out_path, 'w', encoding='utf-8') as fout:
            for paths in path_lists:
                for tmp_path in paths:
                    with open(tmp_path, 'r', encoding='utf-8') as fin:
                        for line in fin:
                            fout.write(line)
                    os.remove(tmp_path)

    # Merge: voice samples first so HF datasets infers list<string> for audios
    print(f'Merging SFT chunks into {sft_output} ...')
    merge_files(sft_output, [sft_voice_paths, sft_text_paths])

    print(f'Merging OPD chunks into {opd_output} ...')
    merge_files(opd_output, [opd_voice_paths, opd_text_paths])

    # Cleanup tmp dir
    tmp_dir = output_dir / 'tmp'
    if tmp_dir.exists():
        try:
            tmp_dir.rmdir()
        except OSError:
            pass

    total_kept = total_voice + total_text
    print(f'\nDone!')
    print(f'  Input:   {total}')
    print(f'  Kept:    {total_kept} ({total_kept / total * 100:.1f}%)')
    print(f'  Skipped: {total_skipped} ({total_skipped / total * 100:.1f}%)')
    print(f'  Voice:   {total_voice}')
    print(f'  Text:    {total_text}')
    print(f'  SFT:     {sft_output}')
    print(f'  OPD:     {opd_output}')


if __name__ == '__main__':
    main()
