# Speechmatics Formatting and Output Features

## Key Formatting Features

### Output Locale

Specify regional spelling conventions for consistency. Available for English variants (GB, US, AU) and Chinese Mandarin (Simplified/Traditional). "Without a specified locale, spelling may be inconsistent within the same transcript."

### Profanity Handling

Supported in English, Italian, and Spanish. Flagged items appear with a `profanity` tag in JSON output for identification or censoring purposes.

### Disfluencies

Hesitation sounds like "um" and "hmm" are automatically tagged in English transcripts. The system can remove them entirely using `remove_disfluencies: true`, which "simplifies client-side processing by removing hesitation sounds."

### Word Replacement

Substitute specific words or regex patterns after transcription for censoring, masking sensitive data, or standardizing terminology. "Word replacement is case-sensitive and applied after transcription is complete."

### Smart Formatting

Automatically converts spoken numbers, dates, currencies into properly formatted text. Enable with `enable_entities: true` to receive detailed entity metadata including `spoken_form` and `written_form` representations with individual timing data.

### Punctuation Control

All languages support punctuation marks appropriate to their conventions. Configure using `punctuation_overrides` to specify permitted marks and sensitivity (0-1 scale).

**Note:** "Disabling punctuation may slightly reduce speaker diarization accuracy."

## Punctuation Settings

All Speechmatics output formats support advanced punctuation.

JSON output places punctuation marks in the results list marked with a type of "punctuation".

Disabling punctuation may slightly harm the accuracy of Speaker Diarization.

### Punctuation Configuration

The sensitivity parameter accepts values from 0 to 1. Higher values produce more punctuation in the output.

The default is 0.5.

The punctuation marks which the client is prepared to accept in transcription output, or the special value 'all' (the default). Unsupported marks are ignored. This value is used to guide the transcription process.

## Accuracy and Quality

### Word Error Rate (WER)

Word Error Rate (WER) is a metric commonly used for benchmarking transcription accuracy.

Punctuation, capitalization and diarization are not measured, but are very important for readability.

Normalization minimizes differences in capitalization, punctuation, and formatting between reference and hypothesis transcripts. This must be done before evaluating the WER, since this can make it harder to see differences in word recognition accuracy.

### Realtime Latency and Accuracy

Partial transcripts accuracy is usually 10-25% lower than the Final transcript. This includes punctuation and capitalization of words.

We recommend experimenting with different settings for the `max_delay` to find the right trade-off between accuracy and latency for your application.

### Speaker Diarization and Punctuation

Speaker diarization uses punctuation to improve accuracy. Small corrections are applied to speaker labels based on sentence boundaries. For example, if the system initially assigns 9 words in a sentence to S1 and 1 word to S2, the lone S2 word may be corrected to S1. This adjustment only works when punctuation is enabled.

Disabling punctuation via the `permitted_marks` setting in `punctuation_overrides` can reduce diarization accuracy. Adjusting punctuation sensitivity can also affect how accurately speakers are identified.

## Preview Mode Improvements

Partial transcripts now support returning full punctuation (., ?, !).

Improved diarization accuracy, especially for low latencies.
