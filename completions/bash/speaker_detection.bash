# Bash completion for speaker_detection CLI
# Source this file or add to ~/.bash_completion.d/

_speaker_detection() {
    local cur prev words cword
    _init_completion || return

    local commands="add list show update delete tag export query enroll embeddings remove-embedding identify verify check-validity validate"

    # Find which subcommand we're in
    local cmd=""
    for ((i=1; i < cword; i++)); do
        case "${words[i]}" in
            add|list|show|update|delete|tag|export|query|enroll|embeddings|remove-embedding|identify|verify|check-validity|validate)
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
        add)
            case "$prev" in
                --name|--nickname|--description)
                    return 0  # User provides value
                    ;;
                --name-context|--metadata)
                    return 0  # User provides KEY=VALUE
                    ;;
                --tag)
                    return 0  # User provides tag
                    ;;
            esac
            case "$cur" in
                -*)
                    COMPREPLY=($(compgen -W "-h --help --name --name-context --nickname --description --tag --metadata" -- "$cur"))
                    ;;
            esac
            ;;
        list)
            case "$prev" in
                --tags|--any-tag)
                    return 0  # User provides tags
                    ;;
                --format)
                    COMPREPLY=($(compgen -W "table json ids" -- "$cur"))
                    return
                    ;;
                --context)
                    return 0  # User provides context
                    ;;
                --limit|--offset)
                    return 0  # User provides number
                    ;;
            esac
            case "$cur" in
                -*)
                    COMPREPLY=($(compgen -W "-h --help --tags --any-tag --format --context --limit --offset" -- "$cur"))
                    ;;
            esac
            ;;
        show)
            case "$prev" in
                --format)
                    COMPREPLY=($(compgen -W "json yaml" -- "$cur"))
                    return
                    ;;
            esac
            case "$cur" in
                -*)
                    COMPREPLY=($(compgen -W "-h --help --format" -- "$cur"))
                    ;;
            esac
            ;;
        update)
            case "$prev" in
                --name|--description|--nickname|--remove-nickname|--tag|--remove-tag)
                    return 0  # User provides value
                    ;;
                --name-context|--metadata)
                    return 0  # User provides KEY=VALUE
                    ;;
            esac
            case "$cur" in
                -*)
                    COMPREPLY=($(compgen -W "-h --help --name --name-context --description --nickname --remove-nickname --tag --remove-tag --metadata" -- "$cur"))
                    ;;
            esac
            ;;
        delete)
            case "$cur" in
                -*)
                    COMPREPLY=($(compgen -W "-h --help --force -f -n --dry-run" -- "$cur"))
                    ;;
            esac
            ;;
        tag)
            case "$prev" in
                --add|--remove)
                    return 0  # User provides tag
                    ;;
            esac
            case "$cur" in
                -*)
                    COMPREPLY=($(compgen -W "-h --help --add --remove" -- "$cur"))
                    ;;
            esac
            ;;
        export)
            case "$prev" in
                --tags|--context)
                    return 0  # User provides value
                    ;;
                --format)
                    COMPREPLY=($(compgen -W "json speechmatics" -- "$cur"))
                    return
                    ;;
                -o|--output)
                    _filedir
                    return
                    ;;
            esac
            case "$cur" in
                -*)
                    COMPREPLY=($(compgen -W "-h --help --tags --context --format -o --output" -- "$cur"))
                    ;;
            esac
            ;;
        query)
            case "$cur" in
                -*)
                    COMPREPLY=($(compgen -W "-h --help" -- "$cur"))
                    ;;
            esac
            ;;
        enroll)
            case "$prev" in
                --backend|-b)
                    COMPREPLY=($(compgen -W "speechmatics" -- "$cur"))
                    return
                    ;;
                --segments|-s)
                    return 0  # User provides segments
                    ;;
                --from-transcript|-t)
                    _filedir json
                    return
                    ;;
                --speaker-label|-l)
                    return 0  # User provides label
                    ;;
            esac
            case "$cur" in
                -*)
                    COMPREPLY=($(compgen -W "-h --help --backend -b --segments -s --from-transcript -t --speaker-label -l --from-stdin -n --dry-run" -- "$cur"))
                    ;;
                *)
                    # Complete audio files
                    _filedir '@(wav|mp3|flac|ogg|m4a|aac)'
                    ;;
            esac
            ;;
        embeddings)
            case "$prev" in
                --backend|-b)
                    COMPREPLY=($(compgen -W "speechmatics" -- "$cur"))
                    return
                    ;;
            esac
            case "$cur" in
                -*)
                    COMPREPLY=($(compgen -W "-h --help --backend -b --show-trust" -- "$cur"))
                    ;;
            esac
            ;;
        remove-embedding)
            case "$cur" in
                -*)
                    COMPREPLY=($(compgen -W "-h --help" -- "$cur"))
                    ;;
            esac
            ;;
        identify)
            case "$prev" in
                --backend|-b)
                    COMPREPLY=($(compgen -W "speechmatics" -- "$cur"))
                    return
                    ;;
                --tags)
                    return 0  # User provides tags
                    ;;
                --threshold)
                    return 0  # User provides number
                    ;;
            esac
            case "$cur" in
                -*)
                    COMPREPLY=($(compgen -W "-h --help --backend -b --tags --threshold" -- "$cur"))
                    ;;
                *)
                    _filedir '@(wav|mp3|flac|ogg|m4a|aac)'
                    ;;
            esac
            ;;
        verify)
            case "$prev" in
                --backend|-b)
                    COMPREPLY=($(compgen -W "speechmatics" -- "$cur"))
                    return
                    ;;
                --threshold)
                    return 0  # User provides number
                    ;;
            esac
            case "$cur" in
                -*)
                    COMPREPLY=($(compgen -W "-h --help --backend -b --threshold" -- "$cur"))
                    ;;
                *)
                    _filedir '@(wav|mp3|flac|ogg|m4a|aac)'
                    ;;
            esac
            ;;
        check-validity)
            case "$cur" in
                -*)
                    COMPREPLY=($(compgen -W "-h --help -v --verbose" -- "$cur"))
                    ;;
            esac
            ;;
        validate)
            case "$cur" in
                -*)
                    COMPREPLY=($(compgen -W "-h --help -v --verbose -q --quiet --strict" -- "$cur"))
                    ;;
            esac
            ;;
    esac
}

complete -F _speaker_detection speaker_detection
