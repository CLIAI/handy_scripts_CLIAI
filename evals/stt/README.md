# STT Evals (Smoke Tests)

Minimal, repeatable audio/text pairs for smoke-testing STT scripts.

## Structure

```
evals/stt/
├── Makefile                 # Builds audio/ from samples/ + voices.txt
├── README.md
├── voices.txt               # One eSpeak-NG voice per line
├── samples/                 # Source text (ground truth)
│   ├── 001-word.test.txt
│   ├── 002-sentence.test.txt
│   └── 003-paragraph.test.txt
├── audio/                   # Generated .ogg files (tracked)
├── run_stt_assemblyai.py    # Runner for ../../stt_assemblyai.py
└── test_all.sh              # Run all eval scripts in this folder
```

## Generate or Refresh Audio

Requires `espeak-ng` and `ffmpeg`:

```bash
cd evals/stt
make
```

This builds `.ogg` files from `samples/*.test.txt` using the voices listed in `voices.txt`.
Intermediate `.wav` files are ignored by git.

## Run the AssemblyAI Smoke Test

Requires `ASSEMBLYAI_API_KEY`:

```bash
cd evals/stt
./run_stt_assemblyai.py
```

The runner normalizes punctuation/case and uses a similarity threshold (default `0.90`).

## Run All Evals

```bash
cd evals/stt
./test_all.sh
```

## Adding Tests

1. Add a new `samples/*.test.txt` file (ASCII text).
2. Run `make` to generate the new `.ogg` files.
3. Commit the `.ogg` files alongside the text.

## Notes

- `../../stt_assemblyai.py` always writes `*.assemblyai.json` next to the audio file; these are ignored by git.
- Use `--keep-json` to retain the JSON files when debugging.
