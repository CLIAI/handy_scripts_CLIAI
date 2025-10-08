#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "requests>=2.31",
#   "replicate>=0.25",
# ]
# requires-python = ">=3.11"
# ///

import argparse
import os
import sys
import requests
import replicate
import re

def normalize_filename(prompt, max_length=64):
    """Normalize prompt to a valid filename."""
    normalized = re.sub(r'[^a-z0-9_-]', '_', prompt.lower())
    return normalized[:max_length]

def check_existing_file(base_name):
    """Check if a file with the given base name (any extension) exists."""
    for ext in ['webp', 'jpg', 'png']:
        if os.path.exists(f"{base_name}.{ext}"):
            return True
    return False

def generate_output_filename(prompt, output):
    """Generate output filename based on prompt or output flag."""
    if output:
        base_name = os.path.splitext(output)[0]
    else:
        base_name = normalize_filename(prompt)
    return base_name

def determine_file_format(content):
    """Determine file format based on content."""
    if content.startswith(b'\x89PNG\r\n\x1a\n'):
        return 'png'
    elif content.startswith(b'\xFF\xD8'):
        return 'jpg'
    elif content.startswith(b'RIFF') and content[8:12] == b'WEBP':
        return 'webp'
    else:
        return 'unknown'

def generate_image(prompt, output_file, verbose=False, force=False):
    """Generate image using Replicate API."""
    if verbose:
        print(f"Generating image for prompt: {prompt}", file=sys.stderr)

    base_name = generate_output_filename(prompt, output_file)

    if not force and check_existing_file(base_name):
        print(f"SKIPPING: File with base name '{base_name}' already exists.", file=sys.stderr)
        sys.exit(1)

    try:
        output = replicate.run(
            "black-forest-labs/flux-schnell",
            input={"prompt": prompt}
        )

        if isinstance(output, list) and len(output) > 0:
            image_url = output[0]
            response = requests.get(image_url)
            response.raise_for_status()
            content = response.content

            ext = determine_file_format(content)
            if ext == 'unknown':
                print("Error: Unable to determine file format.", file=sys.stderr)
                sys.exit(1)

            final_output_file = f"{base_name}.{ext}"
            with open(final_output_file, "wb") as f:
                f.write(content)

            if verbose:
                print(f"Image saved as: {final_output_file}", file=sys.stderr)
            # always print path of enerated image to stdout, so can be consumed by caller of application,
            # and in non verbose mode, on success path it should be only this.
            print(f"{final_output_file}")
        else:
            print("Error: No output received from the API.", file=sys.stderr)
            sys.exit(1)

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Generate image from text prompt using Replicate API")
    parser.add_argument("prompt", nargs="?", help="Text prompt for image generation")
    parser.add_argument("-o", "--output", help="Output filename")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose mode")
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite existing files")
    args = parser.parse_args()

    if not args.prompt and sys.stdin.isatty():
        parser.error("prompt is required if not piping input")

    prompt = args.prompt or sys.stdin.read().strip()

    generate_image(prompt, args.output, args.verbose, args.force)

if __name__ == "__main__":
    main()
