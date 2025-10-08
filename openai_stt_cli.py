#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "numpy>=1.24",
#   "soundfile>=0.12",
#   "sounddevice>=0.4",
#   "prompt_toolkit>=3.0",
#   "openai>=1.0",
# ]
# requires-python = ">=3.11"
# ///
import os
import queue
import tempfile
import time
import argparse
import shutil
import subprocess
import numpy as np
from openai import OpenAI

# Argument parsing
parser = argparse.ArgumentParser(description='OpenAI STT/ASR CLI (Voice to Text)')
parser.add_argument('-s', '--silent', action='store_true', help='Run in silent mode without prompt')
parser.add_argument('-o', '--output', type=str, help='Specify output file for transcript')
parser.add_argument('-a', '--append', type=str, help='Specify output file to append transcript')
parser.add_argument('-c', '--clipboard', action='store_true', help='Copy result to clipboard using xclip')
args = parser.parse_args()

client = OpenAI()

try:
    import soundfile as sf
except (OSError, ModuleNotFoundError):
    sf = None

from prompt_toolkit.shortcuts import prompt

# from .dump import dump  # noqa: F401


class SoundDeviceError(Exception):
    pass


class Voice:
    max_rms = 0
    min_rms = 1e5
    pct = 0

    threshold = 0.15

    def __init__(self):
        if sf is None:
            print("The soundfile module is not available. Please install it using 'pip install soundfile'.")
            raise SoundDeviceError("The soundfile module is required but not installed.")
        try:
            if not args.silent:
                print("Initializing sound device...")
            import sounddevice as sd
            self.sd = sd
        except Exception as e:
            print(f"An error occurred while initializing the sound device: {e}")
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
        return f"Recording, press ENTER when done... {dur:.1f}sec {bar}"

    # This block is responsible for handling the KeyboardInterrupt exception
    # which is triggered by pressing [Ctrl]+[C]. When caught, it simply returns
    # from the function without proceeding to the transcription step.

    def raw_record_and_transcribe(self, history, language):
        self.q = queue.Queue()

        filename = tempfile.mktemp(suffix=".wav")

        sample_rate = 16000  # 16kHz

        self.start_time = time.time()

        with self.sd.InputStream(samplerate=sample_rate, channels=1, callback=self.callback):
            if not args.silent:
                prompt(self.get_prompt, refresh_interval=0.1)
            else:
                #input("Press ENTER to stop recording...")
                input("") # silently wait for ENTER in silent mode.

        # Only proceed with transcription if there are audio frames in the queue
        if not self.q.empty():
            with sf.SoundFile(filename, mode="x", samplerate=sample_rate, channels=1) as file:
                while not self.q.empty():
                    file.write(self.q.get())

            with open(filename, "rb") as fh:
                transcript = client.audio.transcriptions.create(model="whisper-1", file=fh)
            transcript_text = transcript.text
        else:
            transcript_text = "No audio recorded."

        return transcript_text


if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable.")
    # print(Voice().record_and_transcribe()) #original line
    try:
        transcript = Voice().raw_record_and_transcribe(history="", language="en")
        # Handle output based on the silent mode and output file argument
    except KeyboardInterrupt:
        print("\nRecording interrupted by user.")
        exit(0)
    if args.clipboard:
        if shutil.which('xclip'):
            transcript_trimmed = transcript.strip()
            subprocess.run("xclip -selection clipboard", input=transcript_trimmed, text=True, shell=True)

        else:
            print("xclip is not available. Please install xclip or use a different method to copy to clipboard.")
            exit(1)
    if args.output or args.append:
        transcript = transcript.rstrip('\n') + '\n'
        file_mode = 'a' if args.append else 'w'
        output_file = args.append if args.append else args.output
        with open(output_file, file_mode) as f:
            f.write(transcript)
    else:
        print(transcript)
