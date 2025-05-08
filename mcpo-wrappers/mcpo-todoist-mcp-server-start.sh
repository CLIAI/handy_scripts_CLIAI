#!/bin/bash
# verbose mode
set -x
# exit on error
set -e

mkdir -p ~/github/open-webui
cd ~/github/open-webui/
REPO=mcpo

# Ensure that todoist-mcp-server works under `npx todoist-mcp-server` (maybe not installed globally but only for user via `npm install github:abhiz123/todoist-mcp-server`)

# Check if the 'todoist-mcp-server' command works with npx.
# This checks if the package is either installed globally or accessible for the user.
if ! npx todoist-mcp-server --help &>/dev/null; then
    # If 'npx todoist-mcp-server' command fails, proceed with local installation.
    echo "npx todoist-mcp-server is not available. Installing todoist-mcp-server locally from GitHub..."

    # Install the todoist-mcp-server package from the GitHub repository into the current project.
    # This command installs the package in the current directory/node_modules.
    npm install github:abhiz123/todoist-mcp-server

    echo "Installation complete. You can now use './node_modules/.bin/todoist-mcp-server' to run the server."
fi

# Clone the repository if it doesn't exist
if [ ! -d "$REPO" ]; then
    git clone https://github.com/open-webui/$REPO.git
else
    echo "Repository '$REPO' already exists."
fi

uvx mcpo --host 127.0.0.1 --port 7272 -- npx todoist-mcp-server

