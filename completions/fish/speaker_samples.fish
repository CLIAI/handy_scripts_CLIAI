# Fish completion for speaker_samples CLI
# Place in ~/.config/fish/completions/

# Disable file completion by default
complete -c speaker_samples -f

# Global options
complete -c speaker_samples -n "__fish_use_subcommand" -s h -l help -d "Show help message"
complete -c speaker_samples -n "__fish_use_subcommand" -s q -l quiet -d "Suppress status messages"

# Subcommands
complete -c speaker_samples -n "__fish_use_subcommand" -a extract -d "Extract samples from audio"
complete -c speaker_samples -n "__fish_use_subcommand" -a segments -d "Output segment times as JSONL"
complete -c speaker_samples -n "__fish_use_subcommand" -a list -d "List stored samples"
complete -c speaker_samples -n "__fish_use_subcommand" -a info -d "Show sample metadata"
complete -c speaker_samples -n "__fish_use_subcommand" -a remove -d "Remove samples"
complete -c speaker_samples -n "__fish_use_subcommand" -a speakers -d "List speakers in transcript"
complete -c speaker_samples -n "__fish_use_subcommand" -a review -d "Review samples (approve/reject)"

# extract subcommand
complete -c speaker_samples -n "__fish_seen_subcommand_from extract" -s h -l help -d "Show help"
complete -c speaker_samples -n "__fish_seen_subcommand_from extract" -s t -l transcript -d "Transcript JSON file" -rF
complete -c speaker_samples -n "__fish_seen_subcommand_from extract" -s l -l speaker-label -d "Speaker label in transcript" -r
complete -c speaker_samples -n "__fish_seen_subcommand_from extract" -s s -l speaker-id -d "Target speaker ID for storage" -r
complete -c speaker_samples -n "__fish_seen_subcommand_from extract" -l format -d "Output format" -ra "mp3 wav"
complete -c speaker_samples -n "__fish_seen_subcommand_from extract" -l min-duration -d "Minimum segment duration (sec)" -r
complete -c speaker_samples -n "__fish_seen_subcommand_from extract" -l max-gap -d "Max gap to merge segments (sec)" -r
complete -c speaker_samples -n "__fish_seen_subcommand_from extract" -l max-segments -d "Maximum segments to extract" -r
complete -c speaker_samples -n "__fish_seen_subcommand_from extract" -l max-duration -d "Maximum total duration (sec)" -r
complete -c speaker_samples -n "__fish_seen_subcommand_from extract" -s n -l dry-run -d "Show what would be extracted"
complete -c speaker_samples -n "__fish_seen_subcommand_from extract" -s v -l verbose -d "Verbose output"

# segments subcommand
complete -c speaker_samples -n "__fish_seen_subcommand_from segments" -s h -l help -d "Show help"
complete -c speaker_samples -n "__fish_seen_subcommand_from segments" -s t -l transcript -d "Transcript JSON file" -rF
complete -c speaker_samples -n "__fish_seen_subcommand_from segments" -s l -l speaker-label -d "Speaker label" -r
complete -c speaker_samples -n "__fish_seen_subcommand_from segments" -s s -l speaker-id -d "Speaker ID for output" -r
complete -c speaker_samples -n "__fish_seen_subcommand_from segments" -s a -l audio -d "Audio file path (for output)" -rF
complete -c speaker_samples -n "__fish_seen_subcommand_from segments" -l min-duration -d "Minimum segment duration" -r
complete -c speaker_samples -n "__fish_seen_subcommand_from segments" -l max-gap -d "Max gap to merge" -r

# list subcommand
complete -c speaker_samples -n "__fish_seen_subcommand_from list" -s h -l help -d "Show help"
complete -c speaker_samples -n "__fish_seen_subcommand_from list" -l format -d "Output format" -ra "table json"
complete -c speaker_samples -n "__fish_seen_subcommand_from list" -l show-review -d "Show review status"
complete -c speaker_samples -n "__fish_seen_subcommand_from list" -l status -d "Filter by review status" -ra "pending reviewed rejected"
complete -c speaker_samples -n "__fish_seen_subcommand_from list" -l limit -d "Maximum number of results" -r
complete -c speaker_samples -n "__fish_seen_subcommand_from list" -l offset -d "Skip first N results" -r

# info subcommand
complete -c speaker_samples -n "__fish_seen_subcommand_from info" -s h -l help -d "Show help"
complete -c speaker_samples -n "__fish_seen_subcommand_from info" -l format -d "Output format" -ra "yaml json"

# remove subcommand
complete -c speaker_samples -n "__fish_seen_subcommand_from remove" -s h -l help -d "Show help"
complete -c speaker_samples -n "__fish_seen_subcommand_from remove" -l all -d "Remove all samples"
complete -c speaker_samples -n "__fish_seen_subcommand_from remove" -l source -d "Remove samples from matching source path" -r
complete -c speaker_samples -n "__fish_seen_subcommand_from remove" -s f -l force -d "Skip confirmation"
complete -c speaker_samples -n "__fish_seen_subcommand_from remove" -s n -l dry-run -d "Show what would be removed"

# speakers subcommand
complete -c speaker_samples -n "__fish_seen_subcommand_from speakers" -s h -l help -d "Show help"

# review subcommand
complete -c speaker_samples -n "__fish_seen_subcommand_from review" -s h -l help -d "Show help"
complete -c speaker_samples -n "__fish_seen_subcommand_from review" -l source-b3sum -d "Review all samples from source with this b3sum prefix" -r
complete -c speaker_samples -n "__fish_seen_subcommand_from review" -l approve -d "Mark as reviewed/approved"
complete -c speaker_samples -n "__fish_seen_subcommand_from review" -l reject -d "Mark as rejected"
complete -c speaker_samples -n "__fish_seen_subcommand_from review" -l notes -d "Review notes" -r
complete -c speaker_samples -n "__fish_seen_subcommand_from review" -s v -l verbose -d "Verbose output"
