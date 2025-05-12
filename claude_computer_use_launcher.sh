#!/bin/bash

# Just lanuching script will use docker container from Anthropic's repository
# to launch Claude Computer Use, that is running container with GUI graphical
# enviornment and agent performing tasks.

# Handy script to launch Claude Computer Use:
# https://github.com/anthropics/anthropic-quickstarts/tree/main/computer-use-demo

# You can override settings by setting environment variables before running this script.
# e.g.:
# export ANTHROPIC_API_KEY=your_api_key
# export CLAUDE_COMPUTER_USE_VOLUME=your_volume_name
# export CLAUDE_COMPUTER_USE_PORT_VNC=your_vnc_port
# export CLAUDE_COMPUTER_USE_PORT_STREAMLIT=your_streamlit_port
# export CLAUDE_COMPUTER_USE_PORT_DESKTOP_VIEW=your_desktop_view_port
# export CLAUDE_COMPUTER_USE_PORT=your_full_interface_port
# Then run the script:
# ./claude_computer_use_launcher.sh
#
# Or inline:
# ANTHROPIC_API_KEY=your_api_key CLAUDE_COMPUTER_USE_VOLUME=your_volume_name CLAUDE_COMPUTER_USE_PORT_VNC=2222 CLAUDE_COMPUTER_USE_PORT_STREAMLIT=3333 CLAUDE_COMPUTER_USE_PORT_DESKTOP_VIEW=4444 CLAUDE_COMPUTER_USE_PORT=5555 ./claude_computer_use_launcher.sh

if [ -z "$ANTHROPIC_API_KEY" ]; then
  echo "Please set the ANTHROPIC_API_KEY environment variable."
  exit 1
fi

# Set the volume name for Claude Computer Use, if not already set
CLAUDE_COMPUTER_USE_VOLUME="${CLAUDE_COMPUTER_USE_VOLUME:claude-computer-use}"

# Set environment variables for port numbers, use defaults if not provided
CLAUDE_COMPUTER_USE_PORT_VNC="${CLAUDE_COMPUTER_USE_PORT_VNC:-7701}"
CLAUDE_COMPUTER_USE_PORT_STREAMLIT="${CLAUDE_COMPUTER_USE_PORT_STREAMLIT:-7702}"
CLAUDE_COMPUTER_USE_PORT_DESKTOP_VIEW="${CLAUDE_COMPUTER_USE_PORT_DESKTOP_VIEW:-7703}"
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
