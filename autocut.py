#!/usr/bin/env python3
"""Autocut: Cut video to speech-only segments and add subtitles."""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv


FFMPEG_BIN = "ffmpeg"
FFPROBE_BIN = "ffprobe"


def run_ffmpeg(args, desc="ffmpeg"):
    """Run an ffmpeg command, raising on failure."""
    cmd = [FFMPEG_BIN] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR in {desc}:", file=sys.stderr)
        print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr, file=sys.stderr)
        raise RuntimeError(f"{desc} failed with exit code {result.returncode}")
    return result


def format_time(seconds):
    """Format seconds as H:MM:SS.mmm for display."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:06.3f}"


def stage1_extract_audio(input_path, temp_dir, audio_stream=1):
    """Extract mono 16kHz WAV audio for VAD and transcription."""
    print("[Stage 1/6] Extracting audio...")
    t0 = time.time()
    audio_path = os.path.join(temp_dir, "audio_16k_mono.wav")
    run_ffmpeg([
        "-i", input_path,
        "-map", f"0:{audio_stream}",
        "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le",
        "-y", "-loglevel", "warning",
        audio_path,
    ], desc="audio extraction")
    print(f"  Done ({time.time() - t0:.1f}s)")
    return audio_path


def stage2_detect_speech(audio_path, hf_token):
    """Run pyannote VAD to detect speech segments."""
    print("[Stage 2/6] Detecting speech with pyannote...")
    t0 = time.time()

    import torch
    import soundfile as sf
    from pyannote.audio import Model
    from pyannote.audio.pipelines import VoiceActivityDetection

    # cuDNN 9.x (bundled in torch 2.5+) fails on LSTM/RNN ops with driver 535:
    #   "cuDNN error: CUDNN_STATUS_NOT_INITIALIZED"
    # Disabling cuDNN makes PyTorch use its own CUDA LSTM kernels instead.
    # Remove this once the NVIDIA driver is upgraded to 545+.
    torch.backends.cudnn.enabled = False

    model = Model.from_pretrained("pyannote/segmentation-3.0", token=hf_token)
    pipeline = VoiceActivityDetection(segmentation=model)
    pipeline.to(torch.device("cuda"))

    # segmentation-3.0 is a powerset model (onset/offset fixed at 0.5)
    HYPER = {
        "min_duration_on": 0.055,
        "min_duration_off": 0.098,
    }
    pipeline.instantiate(HYPER)

    # Load audio with soundfile to bypass broken torchcodec
    data, sample_rate = sf.read(audio_path, dtype="float32")
    waveform = torch.from_numpy(data).unsqueeze(0)  # (1, num_samples)
    audio_input = {"waveform": waveform, "sample_rate": sample_rate}

    vad_result = pipeline(audio_input)
    segments = [(seg.start, seg.end) for seg in vad_result.get_timeline()]

    total_speech = sum(e - s for s, e in segments)
    print(f"  Found {len(segments)} raw speech segments ({total_speech:.1f}s total)")
    print(f"  Done ({time.time() - t0:.1f}s)")
    return segments


def stage3_merge_segments(segments, merge_gap=0.5, padding=0.25, total_duration=None):
    """Merge nearby segments and add padding."""
    print("[Stage 3/6] Merging segments...")
    if not segments:
        print("  No speech segments found!")
        return []

    # Add padding
    padded = []
    for s, e in segments:
        new_s = max(0, s - padding)
        new_e = e + padding if total_duration is None else min(total_duration, e + padding)
        padded.append((new_s, new_e))

    # Sort and merge
    padded.sort()
    merged = [padded[0]]
    for start, end in padded[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= merge_gap:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    # Filter out segments shorter than 0.5s (too short to encode reliably)
    merged = [(s, e) for s, e in merged if e - s >= 0.5]

    total_speech = sum(e - s for s, e in merged)
    print(f"  {len(merged)} merged segments ({total_speech:.1f}s total)")
    return merged


def _extract_segment(args):
    """Extract a single segment, re-encoding for frame-accurate cuts."""
    idx, start, end, input_path, output_path = args
    duration = end - start
    run_ffmpeg([
        "-ss", str(start), "-i", input_path,
        "-t", str(duration),
        "-map", "0:v:0", "-map", "0:a:0",
        "-c:v", "h264_nvenc", "-preset", "p4", "-cq", "22",
        "-c:a", "aac", "-b:a", "192k",
        "-avoid_negative_ts", "make_zero",
        "-y", "-loglevel", "warning",
        output_path,
    ], desc=f"segment {idx}")
    return idx


def stage4_cut_video(input_path, segments, temp_dir, max_workers=3):
    """Cut video segments and concatenate them."""
    print(f"[Stage 4/6] Cutting {len(segments)} segments (stream copy)...")
    t0 = time.time()

    # Prepare segment extraction tasks
    tasks = []
    for idx, (start, end) in enumerate(segments):
        seg_path = os.path.join(temp_dir, f"segment_{idx:04d}.mp4")
        tasks.append((idx, start, end, input_path, seg_path))

    # Extract segments in parallel
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_extract_segment, task): task[0] for task in tasks}
        for future in as_completed(futures):
            future.result()  # Raise if failed
            completed += 1
            if completed % 20 == 0 or completed == len(tasks):
                print(f"  Extracted {completed}/{len(tasks)} segments")

    # Create concat list
    concat_path = os.path.join(temp_dir, "concat_list.txt")
    with open(concat_path, "w") as f:
        for idx in range(len(segments)):
            seg_path = os.path.join(temp_dir, f"segment_{idx:04d}.mp4")
            f.write(f"file '{seg_path}'\n")

    # Concatenate
    cut_path = os.path.join(temp_dir, "output_cut.mp4")
    run_ffmpeg([
        "-f", "concat", "-safe", "0",
        "-i", concat_path,
        "-c", "copy",
        "-y", "-loglevel", "warning",
        cut_path,
    ], desc="concatenation")

    print(f"  Done ({time.time() - t0:.1f}s)")
    return cut_path


def stage5_transcribe(cut_video_path, temp_dir, whisper_model="large-v3", language="en"):
    """Transcribe cut video and generate SRT."""
    print(f"[Stage 5/6] Transcribing with faster-whisper ({whisper_model})...")
    t0 = time.time()

    # Extract audio from cut video
    cut_audio_path = os.path.join(temp_dir, "cut_audio.wav")
    run_ffmpeg([
        "-i", cut_video_path,
        "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le",
        "-y", "-loglevel", "warning",
        cut_audio_path,
    ], desc="cut audio extraction")

    from faster_whisper import WhisperModel

    model = WhisperModel(whisper_model, device="cuda", compute_type="float16")
    segments_iter, info = model.transcribe(
        cut_audio_path,
        beam_size=5,
        language=language,
        vad_filter=False,
        word_timestamps=True,
    )

    # Build SRT content
    srt_lines = []
    for i, segment in enumerate(segments_iter, 1):
        start_srt = _seconds_to_srt_time(segment.start)
        end_srt = _seconds_to_srt_time(segment.end)
        text = segment.text.strip()
        srt_lines.append(f"{i}\n{start_srt} --> {end_srt}\n{text}\n")

    srt_content = "\n".join(srt_lines)
    srt_path = os.path.join(temp_dir, "output.srt")
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(srt_content)

    print(f"  Generated {len(srt_lines)} subtitle entries")
    print(f"  Done ({time.time() - t0:.1f}s)")
    return srt_path


def _seconds_to_srt_time(seconds):
    """Convert seconds to SRT timestamp format HH:MM:SS,mmm."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def stage6_burn_subtitles(cut_video_path, srt_path, output_path):
    """Burn subtitles into video using libass."""
    print("[Stage 6/6] Burning subtitles...")
    t0 = time.time()

    run_ffmpeg([
        "-i", cut_video_path,
        "-vf", f"subtitles={srt_path}:force_style='FontSize=24,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline=2,Shadow=1'",
        "-c:v", "h264_nvenc", "-preset", "p4", "-cq", "22",
        "-c:a", "copy",
        "-y", "-loglevel", "warning",
        output_path,
    ], desc="subtitle burning")

    print(f"  Done ({time.time() - t0:.1f}s)")


def get_duration(input_path):
    """Get video duration in seconds via ffprobe."""
    result = subprocess.run(
        [FFPROBE_BIN, "-v", "quiet", "-print_format", "json", "-show_format", input_path],
        capture_output=True, text=True,
    )
    import json
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


def main():
    parser = argparse.ArgumentParser(description="Autocut: Cut video to speech segments with subtitles")
    parser.add_argument("input", help="Input video file")
    parser.add_argument("-o", "--output", help="Output filename (default: input_autocut.mp4)")
    parser.add_argument("--audio-stream", type=int, default=1, help="Audio stream index (default: 1)")
    parser.add_argument("--merge-gap", type=float, default=0.5, help="Gap threshold for merging segments (seconds)")
    parser.add_argument("--padding", type=float, default=0.25, help="Padding around speech segments (seconds)")
    parser.add_argument("--whisper-model", default="large-v3", help="Whisper model size (default: large-v3)")
    parser.add_argument("--language", default="en", help="Language code (default: en)")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary files")
    args = parser.parse_args()

    # Load .env
    load_dotenv()
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("ERROR: HF_TOKEN not found. Set it in .env or as an environment variable.", file=sys.stderr)
        print("You need to accept terms at https://huggingface.co/pyannote/segmentation-3.0", file=sys.stderr)
        sys.exit(1)

    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = os.path.abspath(args.output)
    else:
        stem = Path(input_path).stem
        output_path = os.path.join(os.path.dirname(input_path), f"{stem}_autocut.mp4")

    srt_output_path = output_path.rsplit(".", 1)[0] + ".srt"

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print()

    duration = get_duration(input_path)
    print(f"Video duration: {format_time(duration)}")
    print()

    temp_dir = tempfile.mkdtemp(prefix="autocut_")
    try:
        # Stage 1: Extract audio
        audio_path = stage1_extract_audio(input_path, temp_dir, args.audio_stream)

        # Stage 2: VAD
        raw_segments = stage2_detect_speech(audio_path, hf_token)

        # Stage 3: Merge
        merged_segments = stage3_merge_segments(
            raw_segments, args.merge_gap, args.padding, duration
        )
        if not merged_segments:
            print("No speech detected. Exiting.")
            sys.exit(0)

        reduction = (1 - sum(e - s for s, e in merged_segments) / duration) * 100
        print(f"  Removing {reduction:.1f}% of video (non-speech)")
        print()

        # Stage 4: Cut & concat
        cut_path = stage4_cut_video(input_path, merged_segments, temp_dir)

        # Stage 5: Transcribe
        srt_path = stage5_transcribe(cut_path, temp_dir, args.whisper_model, args.language)

        # Stage 6: Burn subtitles
        stage6_burn_subtitles(cut_path, srt_path, output_path)

        # Copy SRT to output location
        shutil.copy2(srt_path, srt_output_path)

        print()
        print(f"Output video: {output_path}")
        print(f"Output SRT:   {srt_output_path}")

        # Show output file size
        out_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"Output size:  {out_size:.1f} MB")

    finally:
        if args.keep_temp:
            print(f"\nTemp files kept at: {temp_dir}")
        else:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
