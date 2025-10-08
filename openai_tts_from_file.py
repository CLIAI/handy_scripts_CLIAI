#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "openai>=1.0.0",
# ]
# requires-python = ">=3.11"
# ///
import sys
import os
from openai import OpenAI

def main():
    if len(sys.argv) == 1:
        sys.stderr.write("Usage: {} <file1> [<file2> ...]\n".format(sys.argv[0]))
        sys.exit(1)

    client = OpenAI()

    for file_path in sys.argv[1:]:
        if os.path.isfile(file_path) and os.access(file_path, os.R_OK):
            sys.stderr.write("Processing file: {}\n".format(file_path))
            with open(file_path, 'r') as file:
                file_contents = file.read()

            response = client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=file_contents,
            )

            base_name, _ = os.path.splitext(file_path)
            output_file = base_name + ".mp3"
            response.stream_to_file(output_file)
            sys.stderr.write("Output saved to: {}\n".format(output_file))
        else:
            sys.stderr.write("Error: File {} is not readable or does not exist.\n".format(file_path))

if __name__ == "__main__":
    main()
