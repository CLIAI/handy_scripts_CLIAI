# Fish completion for speaker_detection CLI
# Place in ~/.config/fish/completions/

# Disable file completion by default
complete -c speaker_detection -f

# Global options
complete -c speaker_detection -n "__fish_use_subcommand" -s h -l help -d "Show help message"
complete -c speaker_detection -n "__fish_use_subcommand" -s q -l quiet -d "Suppress status messages"

# Subcommands
complete -c speaker_detection -n "__fish_use_subcommand" -a add -d "Add a new speaker"
complete -c speaker_detection -n "__fish_use_subcommand" -a list -d "List speakers"
complete -c speaker_detection -n "__fish_use_subcommand" -a show -d "Show speaker details"
complete -c speaker_detection -n "__fish_use_subcommand" -a update -d "Update speaker"
complete -c speaker_detection -n "__fish_use_subcommand" -a delete -d "Delete speaker"
complete -c speaker_detection -n "__fish_use_subcommand" -a tag -d "Manage speaker tags"
complete -c speaker_detection -n "__fish_use_subcommand" -a export -d "Export speakers for STT"
complete -c speaker_detection -n "__fish_use_subcommand" -a query -d "Query with jq expression"
complete -c speaker_detection -n "__fish_use_subcommand" -a enroll -d "Enroll speaker from audio"
complete -c speaker_detection -n "__fish_use_subcommand" -a embeddings -d "List speaker embeddings"
complete -c speaker_detection -n "__fish_use_subcommand" -a remove-embedding -d "Remove an embedding"
complete -c speaker_detection -n "__fish_use_subcommand" -a identify -d "Identify speaker in audio"
complete -c speaker_detection -n "__fish_use_subcommand" -a verify -d "Verify speaker in audio"
complete -c speaker_detection -n "__fish_use_subcommand" -a check-validity -d "Check embedding validity"
complete -c speaker_detection -n "__fish_use_subcommand" -a validate -d "Validate schema of profiles and embeddings"

# add subcommand
complete -c speaker_detection -n "__fish_seen_subcommand_from add" -s h -l help -d "Show help"
complete -c speaker_detection -n "__fish_seen_subcommand_from add" -l name -d "Default display name" -r
complete -c speaker_detection -n "__fish_seen_subcommand_from add" -l name-context -d "Context-specific name (CTX=NAME)" -r
complete -c speaker_detection -n "__fish_seen_subcommand_from add" -l nickname -d "Nickname" -r
complete -c speaker_detection -n "__fish_seen_subcommand_from add" -l description -d "Speaker description" -r
complete -c speaker_detection -n "__fish_seen_subcommand_from add" -l tag -d "Tag" -r
complete -c speaker_detection -n "__fish_seen_subcommand_from add" -l metadata -d "Metadata (KEY=VALUE)" -r

# list subcommand
complete -c speaker_detection -n "__fish_seen_subcommand_from list" -s h -l help -d "Show help"
complete -c speaker_detection -n "__fish_seen_subcommand_from list" -l tags -d "Filter by tags (comma-separated, AND)" -r
complete -c speaker_detection -n "__fish_seen_subcommand_from list" -l any-tag -d "Filter by tags (comma-separated, OR)" -r
complete -c speaker_detection -n "__fish_seen_subcommand_from list" -l format -d "Output format" -ra "table json ids"
complete -c speaker_detection -n "__fish_seen_subcommand_from list" -l context -d "Name context for display" -r
complete -c speaker_detection -n "__fish_seen_subcommand_from list" -l limit -d "Maximum number of results" -r
complete -c speaker_detection -n "__fish_seen_subcommand_from list" -l offset -d "Skip first N results" -r

# show subcommand
complete -c speaker_detection -n "__fish_seen_subcommand_from show" -s h -l help -d "Show help"
complete -c speaker_detection -n "__fish_seen_subcommand_from show" -l format -d "Output format" -ra "json yaml"

# update subcommand
complete -c speaker_detection -n "__fish_seen_subcommand_from update" -s h -l help -d "Show help"
complete -c speaker_detection -n "__fish_seen_subcommand_from update" -l name -d "Update default name" -r
complete -c speaker_detection -n "__fish_seen_subcommand_from update" -l name-context -d "Add/update context-specific name" -r
complete -c speaker_detection -n "__fish_seen_subcommand_from update" -l description -d "Update description" -r
complete -c speaker_detection -n "__fish_seen_subcommand_from update" -l nickname -d "Add nickname" -r
complete -c speaker_detection -n "__fish_seen_subcommand_from update" -l remove-nickname -d "Remove nickname" -r
complete -c speaker_detection -n "__fish_seen_subcommand_from update" -l tag -d "Add tag" -r
complete -c speaker_detection -n "__fish_seen_subcommand_from update" -l remove-tag -d "Remove tag" -r
complete -c speaker_detection -n "__fish_seen_subcommand_from update" -l metadata -d "Add/update metadata (KEY=VALUE)" -r

# delete subcommand
complete -c speaker_detection -n "__fish_seen_subcommand_from delete" -s h -l help -d "Show help"
complete -c speaker_detection -n "__fish_seen_subcommand_from delete" -s f -l force -d "Skip confirmation prompt"
complete -c speaker_detection -n "__fish_seen_subcommand_from delete" -s n -l dry-run -d "Show what would be deleted"

# tag subcommand
complete -c speaker_detection -n "__fish_seen_subcommand_from tag" -s h -l help -d "Show help"
complete -c speaker_detection -n "__fish_seen_subcommand_from tag" -l add -d "Tag to add" -r
complete -c speaker_detection -n "__fish_seen_subcommand_from tag" -l remove -d "Tag to remove" -r

# export subcommand
complete -c speaker_detection -n "__fish_seen_subcommand_from export" -s h -l help -d "Show help"
complete -c speaker_detection -n "__fish_seen_subcommand_from export" -l tags -d "Filter by tags (comma-separated)" -r
complete -c speaker_detection -n "__fish_seen_subcommand_from export" -l context -d "Name context for export" -r
complete -c speaker_detection -n "__fish_seen_subcommand_from export" -l format -d "Export format" -ra "json speechmatics"
complete -c speaker_detection -n "__fish_seen_subcommand_from export" -s o -l output -d "Output file" -rF

# query subcommand
complete -c speaker_detection -n "__fish_seen_subcommand_from query" -s h -l help -d "Show help"

# enroll subcommand
complete -c speaker_detection -n "__fish_seen_subcommand_from enroll" -s h -l help -d "Show help"
complete -c speaker_detection -n "__fish_seen_subcommand_from enroll" -s b -l backend -d "Embedding backend" -ra "speechmatics"
complete -c speaker_detection -n "__fish_seen_subcommand_from enroll" -s s -l segments -d "Time segments (start:end,start:end)" -r
complete -c speaker_detection -n "__fish_seen_subcommand_from enroll" -s t -l from-transcript -d "Extract segments from transcript JSON" -rF
complete -c speaker_detection -n "__fish_seen_subcommand_from enroll" -s l -l speaker-label -d "Speaker label in transcript" -r
complete -c speaker_detection -n "__fish_seen_subcommand_from enroll" -l from-stdin -d "Read segments from stdin"
complete -c speaker_detection -n "__fish_seen_subcommand_from enroll" -s n -l dry-run -d "Show what would be enrolled"

# embeddings subcommand
complete -c speaker_detection -n "__fish_seen_subcommand_from embeddings" -s h -l help -d "Show help"
complete -c speaker_detection -n "__fish_seen_subcommand_from embeddings" -s b -l backend -d "Filter by backend" -ra "speechmatics"
complete -c speaker_detection -n "__fish_seen_subcommand_from embeddings" -l show-trust -d "Show trust level and sample counts"

# remove-embedding subcommand
complete -c speaker_detection -n "__fish_seen_subcommand_from remove-embedding" -s h -l help -d "Show help"

# identify subcommand
complete -c speaker_detection -n "__fish_seen_subcommand_from identify" -s h -l help -d "Show help"
complete -c speaker_detection -n "__fish_seen_subcommand_from identify" -s b -l backend -d "Embedding backend" -ra "speechmatics"
complete -c speaker_detection -n "__fish_seen_subcommand_from identify" -l tags -d "Filter candidates by tags" -r
complete -c speaker_detection -n "__fish_seen_subcommand_from identify" -l threshold -d "Similarity threshold" -r

# verify subcommand
complete -c speaker_detection -n "__fish_seen_subcommand_from verify" -s h -l help -d "Show help"
complete -c speaker_detection -n "__fish_seen_subcommand_from verify" -s b -l backend -d "Embedding backend" -ra "speechmatics"
complete -c speaker_detection -n "__fish_seen_subcommand_from verify" -l threshold -d "Similarity threshold" -r

# check-validity subcommand
complete -c speaker_detection -n "__fish_seen_subcommand_from check-validity" -s h -l help -d "Show help"
complete -c speaker_detection -n "__fish_seen_subcommand_from check-validity" -s v -l verbose -d "Show all embeddings, not just issues"

# validate subcommand
complete -c speaker_detection -n "__fish_seen_subcommand_from validate" -s h -l help -d "Show help"
complete -c speaker_detection -n "__fish_seen_subcommand_from validate" -s v -l verbose -d "Show OK profiles too"
complete -c speaker_detection -n "__fish_seen_subcommand_from validate" -s q -l quiet -d "Only show summary"
complete -c speaker_detection -n "__fish_seen_subcommand_from validate" -l strict -d "Return non-zero exit code on warnings"
