#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "numpy>=1.24",
#   "soundfile>=0.12",
#   "sounddevice>=0.4",
#   "prompt-toolkit>=3.0",
#   "openai>=1.0",
# ]
# requires-python = ">=3.11"
# ///
import os
import sys
import queue
import tempfile
import time
import argparse
import shutil
import subprocess
import numpy as np
from functools import partial
from openai import OpenAI

eprint = partial(print, file=sys.stderr)

# Argument parsing
parser = argparse.ArgumentParser(description='OpenAI STT/ASR CLI (Voice to Text)')
parser.add_argument('-s', '--silent', action='store_true', help='Run in silent mode without prompt')
parser.add_argument('-o', '--output', type=str, help='Specify output file for transcript')
parser.add_argument('-a', '--append', type=str, help='Specify output file to append transcript')
parser.add_argument('-c', '--clipboard', action='store_true', help='Copy result to clipboard using xclip')
parser.add_argument('-x', '--non-interactive', action='store_true', help='Do not prompt user, i.e. use in pipelines (WIP).')
args = parser.parse_args()

client = OpenAI()

try:
    import soundfile as sf
except (OSError, ModuleNotFoundError):
    sf = None

from prompt_toolkit import PromptSession
from prompt_toolkit.output import create_output

# Create an output object that writes to standard error
stderr_output = create_output(sys.stderr)

# Create a PromptSession with the custom output
session = PromptSession(output=stderr_output)

def display_menu(array_of_options=False):
    options = ['1', '2', '3', '']
    if array_of_options:
        return options
    eprint("Recording. Press option and ENTER (or just ENTER for default):")
    eprint("1. transcript with openai API (default)")
    eprint("2. transcript locally with whisper.cpp (with `--speed-up`)")
    eprint("3. transcript locally with whisper.cpp")
    eprint("4. TODO")


class SoundDeviceError(Exception):
    pass


class Voice:
    max_rms = 0
    min_rms = 1e5
    pct = 0

    threshold = 0.15

    def __init__(self):
        if sf is None:
            eprint("The soundfile module is not available. Please install it using 'pip install soundfile'.", file=sys.stderr)
            raise SoundDeviceError("The soundfile module is required but not installed.")
        try:
            import sounddevice as sd
            if not args.silent:
                eprint("Initializing sound device...")
            self.sd = sd
        except Exception as e:
            eprint(f"An error occurred while initializing the sound device: {e}")
            raise SoundDeviceError("Failed to initialize the sound device.")

    def callback(self, indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        rms = np.sqrt(np.mean(indata**2))
        self.max_rms = max(self.max_rms, rms)
        self.min_rms = min(self.min_rms, rms)

        rng = self.max_rms - self.min_rms
        if rng > 0.001:
            self.pct = (rms - self.min_rms) / rng
        else:
            self.pct = 0.5

        self.q.put(indata.copy())

    def get_prompt(self):
        num = 10
        if np.isnan(self.pct) or self.pct < self.threshold:
            cnt = 0
        else:
            cnt = int(self.pct * 10)

        #bar = "░" * cnt + "█" * (num - cnt)
        bar = "#" * cnt + "_" * (num - cnt)
        bar = bar[:num]

        dur = time.time() - self.start_time
        return f"Recording. Press option and ENTER (or just ENTER for default)... {dur:.1f}sec {bar}"

    # This block is responsible for handling the KeyboardInterrupt exception
    # which is triggered by pressing [Ctrl]+[C]. When caught, it simply returns
    # from the function without proceeding to the transcription step.

    def raw_record_only(self):
        self.q = queue.Queue()

        filename = tempfile.mktemp(suffix=".wav")

        sample_rate = 16000  # 16kHz

        self.start_time = time.time()

        with self.sd.InputStream(samplerate=sample_rate, channels=1, callback=self.callback):
            if not args.silent:
                choice = session.prompt(self.get_prompt, refresh_interval=0.1)
            else:
                choice = input("") # silently wait for ENTER in silent mode.

        # Only proceed with transcription if there are audio frames in the queue
        if not self.q.empty():
            with sf.SoundFile(filename, mode="x", samplerate=sample_rate, channels=1) as file:
                while not self.q.empty():
                    file.write(self.q.get())
        else:
            eprint("No audio recorded.")

        return choice, filename

    def transcribe_with_openai_api(self, filename):
        with open(filename, "rb") as fh:
            transcript = client.audio.transcriptions.create(model="whisper-1", file=fh)
        transcript_text = transcript.text

        return transcript_text

    def run_whisper_cpp_in_temp_dir(self, filename, extra_flags=[]):
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                base = os.path.splitext(os.path.basename(filename))[0]
                temp_filename = os.path.join(temp_dir, f"{base}.wav")
                shutil.copyfile(filename, temp_filename)
                eprint(f"Processing {temp_filename} with whisper.cpp in {temp_dir}")
                eprint(f"Base filename: {base}")
                command = ["/usr/bin/time", f"--output={base}.wav.time", "whisper.cpp", "--model", "/usr/share/whisper.cpp-model-large/large.bin"] + extra_flags + ["-otxt", "-ovtt", "-osrt", "-ocsv", temp_filename]
                eprint(f"Running command: {' '.join(command)}")
                result = subprocess.run(command, cwd=temp_dir)
                eprint(f"Command result: {result}")
                with open(f"{temp_dir}/{base}.wav.time", 'r') as f:
                    print(f.read())
                with open(f"{temp_dir}/{base}.wav.txt", 'r') as f:
                    transcript_text = f.read()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
        return transcript_text

    def transcribe_with_whisper_cpp(self, filename, extra_flags=[]):
        return self.run_whisper_cpp_in_temp_dir(filename, extra_flags)

def wait_for_user(transcript, arg_should_not_wait=False):
    if arg_should_not_wait:
        return
    transcript_trimmed = transcript.strip()
    eprint(f"As local transcripts take long, let me wait for you confirming with 'Enter' before exiting. Transcript:\n{transcript_trimmed}")
    input("")


if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable.")
    try:
        if not args.silent:
            display_menu()
        voice = Voice()
        choice, filename = voice.raw_record_only()
        transcript = voice.transcribe_with_openai_api(filename)
    except KeyboardInterrupt:
        eprint("\nRecording interrupted by user.")
        exit(0)
    if not args.silent:
        while choice not in display_menu(array_of_options=True):
            eprint("Invalid choice. Please try again.")
            display_menu()
            choice = input("Enter your choice: ")
    else:
        choice = '1'  # Default to OpenAI API in silent mode
    if choice == '1':
        transcript = voice.transcribe_with_openai_api(filename)
    elif choice == '2':
        transcript = voice.transcribe_with_whisper_cpp(filename, extra_flags=["--speed-up"])
        wait_for_user(transcript, args.non_interactive)
    elif choice == '3':
        transcript = voice.transcribe_with_whisper_cpp(filename, extra_flags=[])
        wait_for_user(transcript, args.non_interactive)
    if args.clipboard:
        if shutil.which('xclip'):
            transcript_trimmed = transcript.strip()
            subprocess.run("xclip -selection clipboard", input=transcript_trimmed, text=True, shell=True)
        else:
            eprint("xclip is not available. Please install xclip or use a different method to copy to clipboard.")
            exit(1)
    if args.output or args.append:
        transcript = transcript.rstrip('\n') + '\n'
        file_mode = 'a' if args.append else 'w'
        output_file = args.append if args.append else args.output
        with open(output_file, file_mode) as f:
            f.write(transcript)
    else:
        print(transcript)
