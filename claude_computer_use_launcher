#!/usr/bin/env bash
# flake8: noqa
#
# This file is a Bash script, not a Python script.
# Recommended usage:
#   bash claude_computer_use_launcher.sh
# or simply rename the file to `.sh` and run it directly.

show_help() {
  cat <<'EOF'
claude_computer_use_launcher.sh - Launch Claude Computer Use in Docker

Purpose:
  This script launches the Anthropic Claude Computer Use demo container, providing a GUI environment
  and agent for task automation via Claude. It sets up the required Docker container, port mappings,
  and persistent volume for configuration and data.

Features:
  - Launches the Claude Computer Use demo container from Anthropic's repository.
  - Exposes the following interfaces on your host:
      * VNC (default: 5900)
      * Streamlit UI (default: 8501)
      * Desktop View (default: 6080)
      * Full Interface (default: 8080)
  - Allows overriding host-side port mappings via environment variables.
  - Persists configuration using a Docker volume (default: claude-computer-use).

Flags / Environment Variables:
  -h, --help
      Show this help message and exit.

  ANTHROPIC_API_KEY
      (Required) Your Anthropic API key.

  CLAUDE_COMPUTER_USE_VOLUME
      Docker volume name for persistent data in /home/computeruse/.anthropic (default: claude-computer-use).

  CLAUDE_COMPUTER_USE_HOME_VOLUME
      Docker volume name for the *entire* /home directory (default: claude-computer-use-home).
      Set to "none" to disable mounting the home directory.

  --reset-config-volume, --reset-volume
      Remove the configuration volume then exit.

  --reset-home-volume
      Remove the home directory volume then exit.

  CLAUDE_COMPUTER_USE_PORT_VNC
      Host port for VNC (default: 5900).

  CLAUDE_COMPUTER_USE_PORT_STREAMLIT
      Host port for Streamlit UI (default: 8501).

  CLAUDE_COMPUTER_USE_PORT_DESKTOP_VIEW
      Host port for Desktop View (default: 6080).

  CLAUDE_COMPUTER_USE_PORT
      Host port for Full Interface (default: 8080).

Example usage:
  export ANTHROPIC_API_KEY=your_api_key
  ./claude_computer_use_launcher.sh

  Or inline:
  ANTHROPIC_API_KEY=your_api_key ./claude_computer_use_launcher.sh

For more information, see:
  https://github.com/anthropics/anthropic-quickstarts/tree/main/computer-use-demo

EOF
}

# Handle CLI flags
RESET_HOME_VOLUME=false
RESET_CONFIG_VOLUME=false

for arg in "$@"; do
  case "$arg" in
    -h|--help)
      show_help
      exit 0
      ;;
    --reset-home-volume)
      RESET_HOME_VOLUME=true
      ;;
    --reset-config-volume|--reset-volume)
      RESET_CONFIG_VOLUME=true
      ;;
  esac
done

# Just lanuching script will use docker container from Anthropic's repository
# to launch Claude Computer Use, that is running container with GUI graphical
# enviornment and agent performing tasks.

# Handy script to launch Claude Computer Use:
# https://github.com/anthropics/anthropic-quickstarts/tree/main/computer-use-demo

# IMPORTANT NOTE ABOUT PORTS:
#
# While this script contains code that *appears* to allow you to override the default ports
# (VNC, Streamlit, Desktop View, Full Interface) by setting environment variables before running,
# the current implementation of the Claude Computer Use container does NOT support changing the
# internal ports it listens on. There is no way to inform the container which ports to use for its
# web UIs or services. The container is hardcoded to use the following ports internally:
#   - VNC: 5900
#   - Streamlit: 8501
#   - Desktop View: 6080
#   - Full Interface: 8080
#
# You *must* use the default port mappings as shown below. Setting the environment variables for
# ports will only change the *host* side of the mapping, but the container will still expect the
# above ports internally. Changing the host ports may break the web UI or other features.
#
# Until Anthropic provides a way to configure the container's internal ports, you should use the
# default ports as shown in this script.
#
# For more details and to track progress on this limitation, see:
#   https://github.com/anthropics/anthropic-quickstarts/issues/271
#
# Example usage:
#   export ANTHROPIC_API_KEY=your_api_key
#   ./claude_computer_use_launcher.sh
#
# Or inline:
#   ANTHROPIC_API_KEY=your_api_key ./claude_computer_use_launcher.sh

if [ -z "$ANTHROPIC_API_KEY" ]; then
  echo "Please set the ANTHROPIC_API_KEY environment variable."
  exit 1
fi

# Volume settings
# Persistent volume for Claude-specific configuration (can be disabled by setting to 'none')
CLAUDE_COMPUTER_USE_VOLUME="${CLAUDE_COMPUTER_USE_VOLUME:-claude-computer-use}"

# Persistent volume for the *entire* home directory of the container user.
# Set CLAUDE_COMPUTER_USE_HOME_VOLUME=none to disable.
CLAUDE_COMPUTER_USE_HOME_VOLUME="${CLAUDE_COMPUTER_USE_HOME_VOLUME:-claude-computer-use-home}"

# Convert 'none' → empty string so we can skip the mount later
if [ "$CLAUDE_COMPUTER_USE_HOME_VOLUME" = "none" ]; then
  CLAUDE_COMPUTER_USE_HOME_VOLUME=""
fi
if [ "$CLAUDE_COMPUTER_USE_VOLUME" = "none" ]; then
  CLAUDE_COMPUTER_USE_VOLUME=""
fi

# Perform cleanup tasks requested by flags
if $RESET_HOME_VOLUME ; then
  echo "Removing home volume: ${CLAUDE_COMPUTER_USE_HOME_VOLUME:-claude-computer-use-home}"
  docker volume rm -f "${CLAUDE_COMPUTER_USE_HOME_VOLUME:-claude-computer-use-home}" 2>/dev/null || true
  exit 0
fi

if $RESET_CONFIG_VOLUME ; then
  echo "Removing config volume: ${CLAUDE_COMPUTER_USE_VOLUME:-claude-computer-use}"
  docker volume rm -f "${CLAUDE_COMPUTER_USE_VOLUME:-claude-computer-use}" 2>/dev/null || true
  exit 0
fi

# Build volume flags array for docker run
DOCKER_VOLUME_FLAGS=()
if [ -n "$CLAUDE_COMPUTER_USE_HOME_VOLUME" ]; then
  DOCKER_VOLUME_FLAGS+=(-v "$CLAUDE_COMPUTER_USE_HOME_VOLUME":/home/computeruse)
fi
if [ -n "$CLAUDE_COMPUTER_USE_VOLUME" ]; then
  DOCKER_VOLUME_FLAGS+=(-v "$CLAUDE_COMPUTER_USE_VOLUME":/home/computeruse/.anthropic)
fi

# Set environment variables for port numbers, use defaults if not provided
# NOTE: See the note above. These variables only affect the host side of the port mapping.
CLAUDE_COMPUTER_USE_PORT_VNC="${CLAUDE_COMPUTER_USE_PORT_VNC:-5900}"
CLAUDE_COMPUTER_USE_PORT_STREAMLIT="${CLAUDE_COMPUTER_USE_PORT_STREAMLIT:-8501}"
CLAUDE_COMPUTER_USE_PORT_DESKTOP_VIEW="${CLAUDE_COMPUTER_USE_PORT_DESKTOP_VIEW:-6080}"
CLAUDE_COMPUTER_USE_PORT="${CLAUDE_COMPUTER_USE_PORT:-8080}"

# (Volume variables are defined above)

# Echo to stderror which ports are being used for what
(
echo "Starting Claude Computer Use with the following settings:"
echo "* Home Volume (entire /home): ${CLAUDE_COMPUTER_USE_HOME_VOLUME:-'<disabled>'}"
echo "* Config Volume (/home/computeruse/.anthropic): ${CLAUDE_COMPUTER_USE_VOLUME:-'<disabled>'}"
echo "* VNC Port: $CLAUDE_COMPUTER_USE_PORT_VNC"
echo "* Streamlit Port: http://$CLAUDE_COMPUTER_USE_PORT_STREAMLIT"
echo "* Desktop View Port: $CLAUDE_COMPUTER_USE_PORT_DESKTOP_VIEW"
echo "* Full Interface Port: http://$CLAUDE_COMPUTER_USE_PORT"
) >&2

# Set Bash verbosity to show commands as they are executed
echo 'set -x'
set -x

# Docker run command
docker run \
  -e ANTHROPIC_API_KEY \
  "${DOCKER_VOLUME_FLAGS[@]}" \
  -p "$CLAUDE_COMPUTER_USE_PORT_VNC":5900 \
  -p "$CLAUDE_COMPUTER_USE_PORT_STREAMLIT":8501 \
  -p "$CLAUDE_COMPUTER_USE_PORT_DESKTOP_VIEW":6080 \
  -p "$CLAUDE_COMPUTER_USE_PORT":8080 \
  -it ghcr.io/anthropics/anthropic-quickstarts:computer-use-demo-latest
