## Personas, Roles... System Messages and Tools

Many tools like:

* mods - <https://github.com/charmbracelet/mods>
* sgpt (Python) - <https://github.com/TheR1D/shell_gpt>
* sgpt (Golang) - <https://github.com/tbckr/sgpt>

allow defining "Personas"/"Roles",
which are actually, system messages for models.

Here in respecitve subdirectories we will collect some example personas/roles for those systmes.

If you want nice collection of those for reuse you can clone and use files from repo:
https://github.com/danielmiessler/fabric/tree/main/patterns

## How to use them?

### mods

`mods --settings` and add in roles section (or roles section itself) with `file:///` path to roles you like:

Here to show format for mods, paths you have to select for your system:

```
roles:
    HAIKU:
        - "file:///path/you/like/for/example/.config/mods/roles/summarize_as_haiku.md.prompt"
    BULLET_POINTS:
        - "file:///path/you/store/roles/bullet_points.prompt"
    TRANSCRIPT_FORMATTER:
        - "file:///path/you/store/roles/format_transcript.md.prompt"
```
