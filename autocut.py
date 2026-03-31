#!/usr/bin/env python3
"""Autocut: Cut video to speech-only segments and add subtitles."""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


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


def _seconds_to_srt_time(seconds):
    """Convert seconds to SRT timestamp format HH:MM:SS,mmm."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def stage1_extract_audio(input_path, temp_dir, audio_stream=1):
    """Extract mono 16kHz WAV audio for whisper."""
    print("[Stage 1/5] Extracting audio...")
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


def stage2_transcribe(audio_path, whisper_model="large-v3", language="en"):
    """Transcribe audio with faster-whisper (uses Silero VAD internally)."""
    print(f"[Stage 2/5] Transcribing with faster-whisper ({whisper_model})...")
    print("  (Silero VAD filters non-speech automatically)")
    t0 = time.time()

    from faster_whisper import WhisperModel

    model = WhisperModel(whisper_model, device="cuda", compute_type="float16")
    segments_iter, info = model.transcribe(
        audio_path,
        beam_size=5,
        language=language,
        vad_filter=True,
        vad_parameters=dict(
            min_speech_duration_ms=250,
            min_silence_duration_ms=500,
            speech_pad_ms=250,
        ),
        word_timestamps=True,
    )

    # Collect segments with their timestamps and text
    segments = []
    for segment in segments_iter:
        segments.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip(),
        })

    total_speech = sum(s["end"] - s["start"] for s in segments)
    print(f"  Found {len(segments)} speech segments ({total_speech:.1f}s total)")
    print(f"  Done ({time.time() - t0:.1f}s)")
    return segments


def merge_speech_segments(segments, merge_gap=0.5, padding=0.25, total_duration=None):
    """Merge nearby speech segments for video cutting."""
    print("[Stage 3/5] Merging segments for cutting...")
    if not segments:
        print("  No speech segments found!")
        return [], []

    # Extract time ranges
    ranges = [(s["start"], s["end"]) for s in segments]

    # Add padding
    padded = []
    for s, e in ranges:
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

    # Filter out segments shorter than 0.5s
    merged = [(s, e) for s, e in merged if e - s >= 0.5]

    total_speech = sum(e - s for s, e in merged)
    print(f"  {len(merged)} merged segments ({total_speech:.1f}s total)")
    return merged


def build_select_expr(segments):
    """Build ffmpeg select/aselect filter expression from segments."""
    parts = [f"between(t,{s:.3f},{e:.3f})" for s, e in segments]
    return "+".join(parts)


def remap_srt(segments, cut_segments, temp_dir):
    """Remap whisper timestamps to the cut timeline and write SRT.

    Uses the same arithmetic as ffmpeg's select+setpts filters:
    the cut timeline is the original timestamps with gaps removed.
    """
    # Precompute cumulative gap before each cut segment
    # cut_time = orig_time - total_gap_before(orig_time)
    gap_before = []
    total_gap = cut_segments[0][0]  # gap before first segment
    gap_before.append((cut_segments[0][0], cut_segments[0][1], total_gap))
    for i in range(1, len(cut_segments)):
        total_gap += cut_segments[i][0] - cut_segments[i - 1][1]
        gap_before.append((cut_segments[i][0], cut_segments[i][1], total_gap))

    def map_time(t):
        for seg_start, seg_end, gap in gap_before:
            if seg_start <= t <= seg_end:
                return t - gap
        return None

    srt_lines = []
    idx = 1
    for seg in segments:
        new_start = map_time(seg["start"])
        new_end = map_time(seg["end"])
        if new_start is not None and new_end is not None and new_end > new_start:
            start_srt = _seconds_to_srt_time(new_start)
            end_srt = _seconds_to_srt_time(new_end)
            srt_lines.append(f"{idx}\n{start_srt} --> {end_srt}\n{seg['text']}\n")
            idx += 1

    srt_content = "\n".join(srt_lines)
    srt_path = os.path.join(temp_dir, "output.srt")
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(srt_content)

    print(f"  Generated {len(srt_lines)} subtitle entries")
    return srt_path


def stage4_cut_video(input_path, segments, temp_dir):
    """Cut video in a single ffmpeg pass using select/aselect filters."""
    print(f"[Stage 4/5] Cutting video (single pass, {len(segments)} segments)...")
    t0 = time.time()

    select_expr = build_select_expr(segments)
    cut_path = os.path.join(temp_dir, "output_cut.mp4")

    run_ffmpeg([
        "-i", input_path,
        "-map", "0:v:0", "-map", "0:a:0",
        "-vf", f"select='{select_expr}',setpts=N/FRAME_RATE/TB",
        "-af", f"aselect='{select_expr}',asetpts=N/SR/TB",
        "-c:v", "h264_nvenc", "-preset", "p4", "-cq", "22",
        "-c:a", "aac", "-b:a", "192k",
        "-y", "-loglevel", "warning",
        cut_path,
    ], desc="single-pass cut")

    print(f"  Done ({time.time() - t0:.1f}s)")
    return cut_path


def stage5_burn_subtitles(cut_video_path, srt_path, output_path):
    """Burn subtitles into video using libass."""
    print("[Stage 5/5] Burning subtitles...")
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

        # Stage 2: Transcribe (with Silero VAD built into faster-whisper)
        segments = stage2_transcribe(audio_path, args.whisper_model, args.language)

        if not segments:
            print("No speech detected. Exiting.")
            sys.exit(0)

        # Stage 3: Merge segments for cutting
        cut_segments = merge_speech_segments(
            segments, args.merge_gap, args.padding, duration
        )
        if not cut_segments:
            print("No segments after merging. Exiting.")
            sys.exit(0)

        reduction = (1 - sum(e - s for s, e in cut_segments) / duration) * 100
        print(f"  Removing {reduction:.1f}% of video (non-speech)")
        print()

        # Build SRT with timestamps remapped to cut timeline
        srt_path = remap_srt(segments, cut_segments, temp_dir)

        # Stage 4: Single-pass cut
        cut_path = stage4_cut_video(input_path, cut_segments, temp_dir)

        # Stage 5: Burn subtitles
        stage5_burn_subtitles(cut_path, srt_path, output_path)

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
