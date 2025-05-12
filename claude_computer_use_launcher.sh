#!/bin/bash

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

# Set the volume name for Claude Computer Use, if not already set
CLAUDE_COMPUTER_USE_VOLUME="${CLAUDE_COMPUTER_USE_VOLUME:claude-computer-use}"

# Set environment variables for port numbers, use defaults if not provided
# NOTE: See the note above. These variables only affect the host side of the port mapping.
CLAUDE_COMPUTER_USE_PORT_VNC="${CLAUDE_COMPUTER_USE_PORT_VNC:-5900}"
CLAUDE_COMPUTER_USE_PORT_STREAMLIT="${CLAUDE_COMPUTER_USE_PORT_STREAMLIT:-8501}"
CLAUDE_COMPUTER_USE_PORT_DESKTOP_VIEW="${CLAUDE_COMPUTER_USE_PORT_DESKTOP_VIEW:-6080}"
CLAUDE_COMPUTER_USE_PORT="${CLAUDE_COMPUTER_USE_PORT:-8080}"

# Volume mapping
CLAUDE_COMPUTER_USE_VOLUME="${CLAUDE_COMPUTER_USE_VOLUME:-claude-computer-use}"

# Echo to stderror which ports are being used for what
(
echo "Starting Claude Computer Use with the following settings:"
echo "* Using volume: $CLAUDE_COMPUTER_USE_VOLUME"
echo "* VNC Port: $CLAUDE_COMPUTER_USE_PORT_VNC"
echo "* Streamlit Port: $CLAUDE_COMPUTER_USE_PORT_STREAMLIT"
echo "* Desktop View Port: $CLAUDE_COMPUTER_USE_PORT_DESKTOP_VIEW"
echo "* Full Interface Port: $CLAUDE_COMPUTER_USE_PORT"
) >&2

# Set Bash verbosity to show commands as they are executed
echo 'set -x'
set -x

# Docker run command
docker run \
  -e ANTHROPIC_API_KEY \
  -v "$CLAUDE_COMPUTER_USE_VOLUME":/home/computeruse/.anthropic \
  -p "$CLAUDE_COMPUTER_USE_PORT_VNC":5900 \
  -p "$CLAUDE_COMPUTER_USE_PORT_STREAMLIT":8501 \
  -p "$CLAUDE_COMPUTER_USE_PORT_DESKTOP_VIEW":6080 \
  -p "$CLAUDE_COMPUTER_USE_PORT":8080 \
  -it ghcr.io/anthropics/anthropic-quickstarts:computer-use-demo-latest
