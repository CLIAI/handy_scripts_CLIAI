# Examples: Multi-Speaker Dialogue Audio Generation

## teacher_student_german_lesson

A bilingual English/German dialogue demonstrating Gemini TTS multi-speaker
capabilities with mixed-language content. An English-speaking teacher quizzes
a student on German phrases.

### Input file

[teacher_student_german_lesson.txt](teacher_student_german_lesson.txt)

### Generated with

```bash
uv run multi-speaker_markup_from_dialog_transcript.py -v \
  -i examples/teacher_student_german_lesson.txt \
  -o examples/teacher_student_german_lesson.mp3 \
  --voices Orus,Aoede \
  -p "Patient, encouraging language teacher with an enthusiastic student. The German words should be pronounced with proper German accent."
```

### Parameters used

* **Model**: `gemini-2.5-flash-tts` (default)
* **Language**: `en-US` (primary language; German phrases handled inline)
* **Voices**: `Orus` (Teacher), `Aoede` (Student)
* **Prompt**: Natural language style direction for delivery tone
* **Output**: MP3, 24kHz, ~47 seconds

### Output

[teacher_student_german_lesson.mp3](teacher_student_german_lesson.mp3) (generated, not checked into git)
