#!/usr/bin/env python3

"""
This script is designed to rescale videos to either 720p or 480p according to
Sora’s recommended aspect ratio matrix (16:9, 3:2, 1:1, 2:3, 9:16). It detects
the original aspect ratio of each input video, then finds the closest standard
Sora aspect ratio and rescales accordingly. By default, the script will produce
an output MP4 using libx264 at a reasonably high quality (CRF=18). It copies
the audio track without re-encoding.

• Works on ArchLinux and Ubuntu with minimal dependencies (only standard Python libraries).
• Accepts multiple input files and processes each in turn.
• Can specify:
  – Resolution to scale to: --resolution/-r [720|480] (default 480)
  – Start time with --ss/-S (float/seconds)
  – End time with --to/-T (float/seconds)
  – Verbosity with --verbose/-v
  – Custom single output name with --output/-o (only valid if exactly one input)
• When used with multiple inputs and no custom output, filenames are suffixed automatically 
  (e.g., input.mp4 → input-480p.mp4). 
• Prints INFO messages to stderr if --verbose/-v is specified.
• May be imported as a module for use in other Python scripts.

Reference for Sora’s 720p and 480p aspect ratio resolutions:
https://www.perplexity.ai/search/sora-720p-480p-resolutions-vs-i3VdN7eNRx69xBhqJv6V2g

  Aspect Ratios and Dimensions for 720p:
      16:9  → 1280×720
      3:2   → 1080×720
      1:1   → 720×720
      2:3   → 480×720
      9:16  → 405×720

  Aspect Ratios and Dimensions for 480p:
      16:9  → 854×480   (commonly used instead of 853×480)
      3:2   → 720×480
      1:1   → 480×480
      2:3   → 320×480
      9:16  → 270×480

"""

import sys
import os
import subprocess
import argparse

ASPECT_RATIOS = {
    "16:9": 16 / 9,
    "3:2": 3 / 2,
    "1:1": 1.0,
    "2:3": 2 / 3,
    "9:16": 9 / 16
}

def log_info(message, verbose=False):
    """
    Print INFO messages to stderr if verbose is True.
    """
    if verbose:
        print(f"INFO: {message}", file=sys.stderr)

def run_cmd(cmd, verbose=False):
    """
    Run a shell command using subprocess.run. Raises an exception on error.
    """
    log_info(f"Executing command: {' '.join(cmd)}", verbose)
    subprocess.run(cmd, check=True)

def probe_video_dimensions(input_file, verbose=False):
    """
    Use ffprobe to return (width, height) of the input video.
    """
    cmd = [
        "ffprobe", 
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=s=,:p=0", 
        input_file
    ]
    
    log_info(f"Probing video dimensions for {input_file}", verbose)
    try:
        result = subprocess.run(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            check=True,
            text=True
        )
        width_str, height_str = result.stdout.strip().split(',')
        width = int(width_str)
        height = int(height_str)
        return width, height
    except subprocess.CalledProcessError as e:
        print(f"ERROR: ffprobe failed for file {input_file}:\n{e.stderr}", file=sys.stderr)
        sys.exit(1)
    except ValueError:
        print(f"ERROR: Could not parse video dimensions for file {input_file}.", file=sys.stderr)
        sys.exit(1)

def find_closest_aspect_ratio(width, height):
    """
    Given a video width and height, return the label of the closest standard 
    aspect ratio from ASPECT_RATIOS.
    """
    if height == 0:
        return "16:9"  # fallback
    actual_ratio = width / height
    best_label = None
    smallest_diff = float("inf")
    for label, ratio in ASPECT_RATIOS.items():
        diff = abs(actual_ratio - ratio)
        if diff < smallest_diff:
            smallest_diff = diff
            best_label = label
    return best_label

def compute_scale_dimensions(aspect_label, target_height, verbose=False):
    """
    Compute the (width, height) for the final scale based on Sora's specs.

    For 16:9 with 480p, we explicitly use 854 for the width.
    Otherwise, we do: width = round(ratio * target_height).
    """
    ratio = ASPECT_RATIOS[aspect_label]
    if aspect_label == "16:9" and target_height == 480:
        final_width = 854
    else:
        final_width = round(ratio * target_height)
    return final_width, target_height

def build_ffmpeg_command(input_file, output_file, width, height, ss=None, to=None, verbose=False):
    """
    Build the ffmpeg command to re-encode the video with the specified scale,
    optionally including -ss and -to for trimming.
    """
    scale_str = f"scale={width}:{height}"
    cmd = ["ffmpeg"]
    
    if ss is not None:
        cmd.extend(["-ss", str(ss)])
    if to is not None:
        cmd.extend(["-to", str(to)])
    
    cmd.extend([
        "-i", input_file,
        "-vf", scale_str,
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "slow",
        "-c:a", "copy",
        output_file
    ])
    
    return cmd

def process_video(input_file, resolution, ss, to, output_file=None, verbose=False):
    """
    Main logic to:
      1. Probe the video for dimensions
      2. Find the closest aspect ratio
      3. Compute final scale
      4. Construct an ffmpeg command
      5. Execute the command
    """
    if not os.path.isfile(input_file):
        print(f"ERROR: Input file {input_file} does not exist.", file=sys.stderr)
        return

    log_info(f"Processing file: {input_file}", verbose)
    in_width, in_height = probe_video_dimensions(input_file, verbose=verbose)
    log_info(f"Original dimensions: {in_width}x{in_height}", verbose)

    aspect_label = find_closest_aspect_ratio(in_width, in_height)
    log_info(f"Closest aspect ratio label: {aspect_label}", verbose)

    # Convert resolution arg to integer height (default to 480 if invalid)
    target_height = 720 if resolution == 720 else 480
    final_width, final_height = compute_scale_dimensions(aspect_label, target_height, verbose)
    log_info(f"Rescaling to: {final_width}x{final_height}", verbose)

    if output_file is None:
        # Generate output file name automatically
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}-{final_height}p.mp4"

    log_info(f"Output file: {output_file}", verbose)

    cmd = build_ffmpeg_command(
        input_file, output_file,
        final_width, final_height,
        ss=ss, to=to,
        verbose=verbose
    )
    run_cmd(cmd, verbose)

def parse_args(argv=None):
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Rescale videos to 720p or 480p fitting Sora requirements. "
                    "Detects original ratio and finds best fit. Re-encodes to MP4."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input video file(s) to process."
    )
    parser.add_argument(
        "-r", "--resolution",
        type=int,
        choices=[480, 720],
        default=480,
        help="Target vertical resolution (480 or 720). Default: 480"
    )
    parser.add_argument(
        "-S", "--ss",
        type=float,
        default=None,
        help="Start time in seconds (float) for trimming (like ffmpeg -ss)."
    )
    parser.add_argument(
        "-T", "--to",
        type=float,
        default=None,
        help="End time in seconds (float) for trimming (like ffmpeg -to)."
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output file name. Only valid if exactly one input file is provided."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging to stderr."
    )
    return parser.parse_args(argv)

def main():
    args = parse_args()
    verbose = args.verbose

    # Validate output usage
    if args.output and len(args.inputs) > 1:
        print("ERROR: --output/-o can only be used with exactly one input file.", file=sys.stderr)
        sys.exit(1)

    for idx, in_file in enumerate(args.inputs):
        # If user specified output and there's only one input, use that. Otherwise None.
        current_out = args.output if (args.output and idx == 0) else None
        
        process_video(
            input_file=in_file,
            resolution=args.resolution,
            ss=args.ss,
            to=args.to,
            output_file=current_out,
            verbose=verbose
        )

if __name__ == "__main__":
    main()

