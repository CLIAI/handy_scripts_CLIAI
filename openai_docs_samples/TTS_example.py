#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "openai>=1.0.0",
# ]
# requires-python = ">=3.11"
# ///
from openai import OpenAI

client = OpenAI()

response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input="Hello world! This is a streaming test.",
)

response.stream_to_file("output.mp3")
