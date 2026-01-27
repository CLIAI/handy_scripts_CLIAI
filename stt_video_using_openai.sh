#!/bin/bash

BACKEND="openai"

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

# Parse flags
STATUS_MODE=false
JSON_OUTPUT=false
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    --status)
      STATUS_MODE=true
      shift
      ;;
    --json)
      JSON_OUTPUT=true
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [--status] [--json] video_file [language_code]" >&2
      echo >&2
      echo "Options:" >&2
      echo "  --status    Check if transcript exists without processing" >&2
      echo "  --json      Output status in JSON format" >&2
      echo >&2
      echo "Arguments:" >&2
      echo "  video_file: The path to the video file you want to transcribe." >&2
      echo "  language_code: Language code (default: auto). Examples: en, de, fr, es, ja" >&2
      echo >&2
      echo "Note: OpenAI Whisper API does not support speaker diarization with whisper-1." >&2
      echo "For speaker diarization, use stt_video_using_assemblyai.sh or stt_video_using_speechmatics.sh" >&2
      exit 0
      ;;
    *)
      POSITIONAL_ARGS+=("$1")
      shift
      ;;
  esac
done

set -- "${POSITIONAL_ARGS[@]}"

if [ $# -eq 0 ]; then
  echo "Usage: $0 [--status] [--json] video_file [language_code]" >&2
  exit 1
fi

VIDEO="$1"
if [ ! -f "$VIDEO" ]; then
  if [ "$STATUS_MODE" = true ]; then
    if [ "$JSON_OUTPUT" = true ]; then
      echo "{\"audio_path\": \"$VIDEO\", \"audio_exists\": false, \"mp3_path\": null, \"mp3_exists\": false, \"transcript_path\": null, \"transcript_exists\": false, \"backend\": \"$BACKEND\"}"
    else
      echo "Audio: $VIDEO (not found)"
      echo "MP3: not found"
      echo "Transcript: not found"
    fi
    exit 0
  else
    echo "Video file $VIDEO does not exist." >&2
    exit 1
  fi
fi

MP3="$VIDEO".mp3
# New naming convention: include backend suffix
TXT_NEW="$MP3"."$BACKEND".txt
# Legacy naming for backward compatibility
TXT_LEGACY="$MP3".txt

# Determine which transcript file to use (prefer new, fallback to legacy)
if [ -f "$TXT_NEW" ]; then
  TXT="$TXT_NEW"
  TXT_IS_LEGACY=false
elif [ -f "$TXT_LEGACY" ]; then
  TXT="$TXT_LEGACY"
  TXT_IS_LEGACY=true
else
  TXT="$TXT_NEW"
  TXT_IS_LEGACY=false
fi

# Status mode: just report what exists
if [ "$STATUS_MODE" = true ]; then
  MP3_EXISTS=false
  MP3_SIZE=0
  if [ -f "$MP3" ]; then
    MP3_EXISTS=true
    MP3_SIZE=$(stat -c%s "$MP3" 2>/dev/null || stat -f%z "$MP3" 2>/dev/null || echo 0)
  fi

  TXT_EXISTS=false
  TXT_SIZE=0
  TXT_PATH="$TXT_NEW"
  if [ -f "$TXT_NEW" ]; then
    TXT_EXISTS=true
    TXT_SIZE=$(stat -c%s "$TXT_NEW" 2>/dev/null || stat -f%z "$TXT_NEW" 2>/dev/null || echo 0)
    TXT_PATH="$TXT_NEW"
    TXT_IS_LEGACY=false
  elif [ -f "$TXT_LEGACY" ]; then
    TXT_EXISTS=true
    TXT_SIZE=$(stat -c%s "$TXT_LEGACY" 2>/dev/null || stat -f%z "$TXT_LEGACY" 2>/dev/null || echo 0)
    TXT_PATH="$TXT_LEGACY"
    TXT_IS_LEGACY=true
  fi

  if [ "$JSON_OUTPUT" = true ]; then
    cat <<EOF
{
  "audio_path": "$VIDEO",
  "audio_exists": true,
  "mp3_path": "$MP3",
  "mp3_exists": $MP3_EXISTS,
  "mp3_size_bytes": $MP3_SIZE,
  "transcript_path": "$TXT_PATH",
  "transcript_exists": $TXT_EXISTS,
  "transcript_size_bytes": $TXT_SIZE,
  "transcript_legacy": $TXT_IS_LEGACY,
  "backend": "$BACKEND"
}
EOF
  else
    echo "Audio: $VIDEO"
    if [ "$MP3_EXISTS" = true ]; then
      echo "MP3: exists ($MP3) - $(numfmt --to=iec $MP3_SIZE 2>/dev/null || echo "$MP3_SIZE bytes")"
    else
      echo "MP3: not found"
    fi
    if [ "$TXT_EXISTS" = true ]; then
      LEGACY_NOTE=""
      if [ "$TXT_IS_LEGACY" = true ]; then
        LEGACY_NOTE=" [legacy naming]"
      fi
      echo "Transcript: exists ($TXT_PATH)$LEGACY_NOTE - $(numfmt --to=iec $TXT_SIZE 2>/dev/null || echo "$TXT_SIZE bytes")"
    else
      echo "Transcript: not found"
    fi
  fi
  exit 0
fi

# Normal transcription mode
LANGUAGE="${2:-}"
if [ -z "$LANGUAGE" ]; then
  read -p 'Language code [auto]:' LANGUAGE
  LANGUAGE="${LANGUAGE:-auto}"
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

# Extract MP3 if needed
extract_mp3 "$VIDEO" "$MP3"

# Check file size (OpenAI limit is 25MB)
FILE_SIZE=$(stat -c%s "$MP3" 2>/dev/null || stat -f%z "$MP3" 2>/dev/null)
MAX_SIZE=$((25 * 1024 * 1024))
if [ "$FILE_SIZE" -gt "$MAX_SIZE" ]; then
  echo "WARNING: Audio file is larger than 25MB (OpenAI limit)." >&2
  echo "Consider splitting the file or using AssemblyAI/Speechmatics instead." >&2
fi

# Use new naming convention for output
TXT="$TXT_NEW"

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
