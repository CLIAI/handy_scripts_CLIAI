# Bash completion for speaker_samples CLI
# Source this file or add to ~/.bash_completion.d/

_speaker_samples() {
    local cur prev words cword
    _init_completion || return

    local commands="extract segments list info remove speakers review"

    # Find which subcommand we're in
    local cmd=""
    for ((i=1; i < cword; i++)); do
        case "${words[i]}" in
            extract|segments|list|info|remove|speakers|review)
                cmd="${words[i]}"
                break
                ;;
        esac
    done

    # If no subcommand yet, complete subcommands
    if [[ -z "$cmd" ]]; then
        case "$cur" in
            -*)
                COMPREPLY=($(compgen -W "-h --help -q --quiet" -- "$cur"))
                ;;
            *)
                COMPREPLY=($(compgen -W "$commands" -- "$cur"))
                ;;
        esac
        return
    fi

    # Subcommand-specific completions
    case "$cmd" in
        extract)
            case "$prev" in
                -t|--transcript)
                    _filedir json
                    return
                    ;;
                -l|--speaker-label|-s|--speaker-id)
                    return 0  # User provides value
                    ;;
                --format)
                    COMPREPLY=($(compgen -W "mp3 wav" -- "$cur"))
                    return
                    ;;
                --min-duration|--max-gap|--max-segments|--max-duration)
                    return 0  # User provides number
                    ;;
            esac
            case "$cur" in
                -*)
                    COMPREPLY=($(compgen -W "-h --help -t --transcript -l --speaker-label -s --speaker-id --format --min-duration --max-gap --max-segments --max-duration -n --dry-run -v --verbose" -- "$cur"))
                    ;;
                *)
                    _filedir '@(wav|mp3|flac|ogg|m4a|aac)'
                    ;;
            esac
            ;;
        segments)
            case "$prev" in
                -t|--transcript)
                    _filedir json
                    return
                    ;;
                -l|--speaker-label|-s|--speaker-id)
                    return 0  # User provides value
                    ;;
                -a|--audio)
                    _filedir '@(wav|mp3|flac|ogg|m4a|aac)'
                    return
                    ;;
                --min-duration|--max-gap)
                    return 0  # User provides number
                    ;;
            esac
            case "$cur" in
                -*)
                    COMPREPLY=($(compgen -W "-h --help -t --transcript -l --speaker-label -s --speaker-id -a --audio --min-duration --max-gap" -- "$cur"))
                    ;;
            esac
            ;;
        list)
            case "$prev" in
                --format)
                    COMPREPLY=($(compgen -W "table json" -- "$cur"))
                    return
                    ;;
                --status)
                    COMPREPLY=($(compgen -W "pending reviewed rejected" -- "$cur"))
                    return
                    ;;
                --limit|--offset)
                    return 0  # User provides number
                    ;;
            esac
            case "$cur" in
                -*)
                    COMPREPLY=($(compgen -W "-h --help --format --show-review --status --limit --offset" -- "$cur"))
                    ;;
            esac
            ;;
        info)
            case "$prev" in
                --format)
                    COMPREPLY=($(compgen -W "yaml json" -- "$cur"))
                    return
                    ;;
            esac
            case "$cur" in
                -*)
                    COMPREPLY=($(compgen -W "-h --help --format" -- "$cur"))
                    ;;
            esac
            ;;
        remove)
            case "$prev" in
                --source)
                    return 0  # User provides source path
                    ;;
            esac
            case "$cur" in
                -*)
                    COMPREPLY=($(compgen -W "-h --help --all --source -f --force -n --dry-run" -- "$cur"))
                    ;;
            esac
            ;;
        speakers)
            case "$cur" in
                -*)
                    COMPREPLY=($(compgen -W "-h --help" -- "$cur"))
                    ;;
                *)
                    _filedir json
                    ;;
            esac
            ;;
        review)
            case "$prev" in
                --source-b3sum|--notes)
                    return 0  # User provides value
                    ;;
            esac
            case "$cur" in
                -*)
                    COMPREPLY=($(compgen -W "-h --help --source-b3sum --approve --reject --notes -v --verbose" -- "$cur"))
                    ;;
            esac
            ;;
    esac
}

complete -F _speaker_samples speaker_samples
