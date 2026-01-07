#!/bin/bash

if ! command -v stt_openai.py &> /dev/null
then
    echo 'The script stt_openai.py is required to run this program.
    It is not currently in your PATH.
    Please ensure that it is available.
    One way to do this is by cloning the repository
    https://github.com/CLIAI/handy_scripts
    into a directory in your PATH.' >&2
    exit 1
fi

if [ $# -eq 0 ] || [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
  echo "Usage: $0 video_file [language_code]" >&2
  echo >&2
  echo "* video_file: The path to the video file you want to transcribe." >&2
  echo "* language_code: Language code (default: auto). Examples: en, de, fr, es, ja" >&2
  echo >&2
  echo "Note: OpenAI Whisper API does not support speaker diarization with whisper-1." >&2
  echo "For speaker diarization, use stt_video_using_assemblyai.sh or stt_video_using_speechmatics.sh" >&2
  exit 1
fi

function extract_mp3() {
  local video="$1"
  local MP3="$2"
  if [ ! -f "$MP3" ]; then
    set -x
    ffmpeg -i "$video" -vn -ab 128k -ar 44100 -y "$MP3"
    set +x
  else
    echo "File $MP3 already exists." >&2
  fi
}

function transcribe() {
  local LANGUAGE="$1"
  local MP3="$2"
  local TXT="$3"

  # Always call stt_openai.py - it will handle idempotence internally
  if [ "$LANGUAGE" == "auto" ]; then
    set -x
    stt_openai.py -o "$TXT" "$MP3"
  else
    set -x
    stt_openai.py -l "$LANGUAGE" -o "$TXT" "$MP3"
  fi
  set +x
}

if [ $# -eq 0 ]; then
  echo "Usage: $0 video_file [language_code]" >&2
  exit 0
fi

VIDEO="$1"
if [ ! -f "$VIDEO" ]; then
  echo "Video file $VIDEO does not exist." >&2
  exit 1
fi

MP3="$VIDEO".mp3
TXT="$MP3".txt

LANGUAGE="${2:-}"
if [ -z "$LANGUAGE" ]; then
  read -p 'Language code [auto]:' LANGUAGE
  LANGUAGE="${LANGUAGE:-auto}"
fi

# Extract MP3 if needed
extract_mp3 "$VIDEO" "$MP3"

# Check file size (OpenAI limit is 25MB)
FILE_SIZE=$(stat -c%s "$MP3" 2>/dev/null || stat -f%z "$MP3" 2>/dev/null)
MAX_SIZE=$((25 * 1024 * 1024))
if [ "$FILE_SIZE" -gt "$MAX_SIZE" ]; then
  echo "WARNING: Audio file is larger than 25MB (OpenAI limit)." >&2
  echo "Consider splitting the file or using AssemblyAI/Speechmatics instead." >&2
fi

# Transcribe using OpenAI
# stt_openai.py will handle idempotence - if transcript exists, it will display it
transcribe "$LANGUAGE" "$MP3" "$TXT"

# Display a message indicating completion
if [ -f "$TXT" ]; then
  echo "Transcript is available at: $TXT" >&2

  # Only print the transcript if it wasn't already printed by stt_openai.py
  # We can check if the transcript was just created by comparing timestamps
  if [ -s "$TXT" ] && [ "$TXT" -ot "$MP3" ]; then
    cat "$TXT"
  fi
fi
