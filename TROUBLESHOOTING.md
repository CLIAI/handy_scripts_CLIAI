# Troubleshooting Guide

Common errors and solutions for speaker detection tools.

## Installation & Dependencies

### Error: `ffmpeg not found`

```
FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'
```

**Solution**: Install ffmpeg for your system:

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# Arch Linux
sudo pacman -S ffmpeg

# macOS
brew install ffmpeg

# Verify installation
ffmpeg -version
```

### Error: `ModuleNotFoundError: No module named 'yaml'`

```
ModuleNotFoundError: No module named 'yaml'
```

**Solution**: Install PyYAML:

```bash
pip install pyyaml
```

### Error: `ModuleNotFoundError: No module named 'requests'`

**Solution**: Install requests:

```bash
pip install requests
```

---

## API & Authentication

### Error: `SPEECHMATICS_API_KEY not set`

```
Error: SPEECHMATICS_API_KEY environment variable not set
```

**Solution**: Set your Speechmatics API key:

```bash
export SPEECHMATICS_API_KEY="your-api-key-here"

# Or add to shell profile (~/.bashrc, ~/.zshrc)
echo 'export SPEECHMATICS_API_KEY="your-key"' >> ~/.bashrc
source ~/.bashrc
```

### Error: `401 Unauthorized` or `Invalid API key`

```
Error: API request failed: 401 Unauthorized
```

**Causes**:

* API key is expired or revoked
* API key copied incorrectly (extra spaces/newlines)
* Wrong API key for region

**Solution**:

```bash
# Check your key (shows first/last few chars)
echo "${SPEECHMATICS_API_KEY:0:8}...${SPEECHMATICS_API_KEY: -4}"

# Verify it works with a test request
curl -H "Authorization: Bearer $SPEECHMATICS_API_KEY" \
     https://asr.api.speechmatics.com/v2/jobs
```

### Error: `Connection timed out` or `Network unreachable`

**Causes**:

* No internet connection
* Firewall blocking Speechmatics API
* Corporate proxy not configured

**Solution**:

```bash
# Test connectivity
curl -v https://asr.api.speechmatics.com/v2/jobs

# If behind proxy
export HTTPS_PROXY=http://your-proxy:port
```

---

## Audio & Transcription

### Error: `Audio too short for enrollment`

```
Error: Audio segment too short (0.3s). Minimum required: 0.5s
```

**Solution**: Use longer audio segments or adjust minimum duration:

```bash
# Extract with longer minimum duration
./speaker_samples extract audio.mp3 \
    -t transcript.json \
    -l S1 \
    -s alice \
    --min-duration 1.0
```

### Error: `No speakers detected in audio`

```
ValueError: No speakers detected in audio. Ensure audio contains clear speech.
```

**Causes**:

* Audio is too quiet or noisy
* Audio format not supported
* Wrong sample rate

**Solution**:

```bash
# Check audio file properties
ffprobe -v quiet -print_format json -show_streams audio.mp3

# Convert to known working format (16kHz mono WAV)
ffmpeg -i audio.mp3 -ac 1 -ar 16000 audio_converted.wav

# Retry with converted file
./speaker_detection enroll alice audio_converted.wav ...
```

### Error: `Unsupported audio format`

**Solution**: Convert to supported format:

```bash
# Convert to WAV
ffmpeg -i input.m4a -acodec pcm_s16le -ar 16000 output.wav

# Supported formats: WAV, MP3, FLAC, OGG
```

---

## Transcript Format Issues

### Error: `Unknown transcript format`

```
Error: Could not detect transcript format. Expected Speechmatics or AssemblyAI.
```

**Causes**:

* Transcript from unsupported provider
* Malformed JSON
* Wrong file provided

**Solution**:

```bash
# Check transcript structure
jq 'keys' transcript.json

# Speechmatics format has "results" array:
jq '.results[0] | keys' transcript.json
# Output: ["alternatives", "end_time", "start_time", "type"]

# AssemblyAI format has "utterances" array:
jq '.utterances[0] | keys' transcript.json
# Output: ["confidence", "end", "speaker", "start", "text", "words"]
```

### Error: `Speaker label not found in transcript`

```
Error: Speaker 'Alice' not found. Available speakers: S1, S2, S3
```

**Solution**: Use the correct speaker label from the transcript:

```bash
# List available speakers
./speaker_samples speakers transcript.json
# Output: Speakers: S1, S2, S3

# Use exact label
./speaker_samples extract audio.mp3 -t transcript.json -l S1 -s alice
```

---

## Data & Storage

### Error: `Speaker not found`

```
Error: Speaker 'alice' not found
```

**Causes**:

* Speaker never created
* Looking in wrong data directory
* Typo in speaker ID

**Solution**:

```bash
# List existing speakers
./speaker_detection list

# Check data directory
echo $SPEAKERS_EMBEDDINGS_DIR
ls ~/.config/speakers_embeddings/db/

# Create if missing
./speaker_detection add alice --name "Alice"
```

### Error: `Permission denied` on data directory

```
PermissionError: [Errno 13] Permission denied: '/home/user/.config/speakers_embeddings/db'
```

**Solution**:

```bash
# Fix permissions
chmod -R u+rw ~/.config/speakers_embeddings/

# Or use different directory
export SPEAKERS_EMBEDDINGS_DIR=/tmp/my_speakers
```

### Error: `Profile/metadata file corrupted`

```
JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```

**Solution**:

```bash
# Check file content
cat ~/.config/speakers_embeddings/db/alice.json

# If corrupted, remove and recreate
rm ~/.config/speakers_embeddings/db/alice.json
./speaker_detection add alice --name "Alice"

# Re-extract samples and re-enroll
```

### Data disappeared: Where are my profiles?

**Solution**:

```bash
# Check current data directory
echo "${SPEAKERS_EMBEDDINGS_DIR:-~/.config/speakers_embeddings}"

# List all profiles
ls -la ~/.config/speakers_embeddings/db/

# If using custom directory, check that
ls -la $SPEAKERS_EMBEDDINGS_DIR/db/
```

---

## Trust & Validity Issues

### Warning: `INVALIDATED embedding detected`

```
INVALIDATED: alice - emb-abc123 (1 rejected sample)
```

**Cause**: A sample used for enrollment was later rejected.

**Solution**:

```bash
# Check which samples are rejected
./speaker_samples list alice --status rejected

# Re-enroll with only approved samples
./speaker_detection enroll alice new_audio.mp3 \
    --from-transcript transcript.json \
    --speaker-label S1
```

### Warning: `Trust level: low`

```
emb-abc123  2026-01-16  [low] (0r/3u/0x)
```

**Cause**: All samples are unreviewed.

**Solution**:

```bash
# Review and approve samples
./speaker_samples list alice --show-review
./speaker_samples review alice sample-001 --approve
./speaker_samples review alice sample-002 --approve

# Check validity again
./speaker_detection check-validity
```

---

## Testing & Development

### Error: `Test audio files not found`

```
Missing test audio: evals/speaker_detection/audio/test_001-two-speakers.wav
```

**Solution**:

```bash
# Generate test audio files
cd evals/speaker_detection
make all

# Verify files exist
ls -la audio/
```

### Running tests in isolated environment

```bash
# Use temporary directory for tests
export SPEAKERS_EMBEDDINGS_DIR=$(mktemp -d)

# Run tests
./evals/speaker_detection/test_all.sh

# Clean up
rm -rf $SPEAKERS_EMBEDDINGS_DIR
```

### Docker testing

```bash
# Build and run tests in Docker
./evals/run_docker_tests.sh

# Interactive debugging
./evals/run_docker_tests.sh --shell
```

---

## Still Stuck?

1. **Check verbose output**: Add `-v` or `--verbose` to commands for more details
2. **Check file permissions**: Ensure read/write access to data directory
3. **Verify dependencies**: Run `python3 -c "import yaml; import requests; print('OK')"`
4. **Try fresh directory**: `export SPEAKERS_EMBEDDINGS_DIR=/tmp/test && ./speaker_detection list`
5. **Open an issue**: [GitHub Issues](https://github.com/CLIAI/handy_scripts_CLIAI/issues)
