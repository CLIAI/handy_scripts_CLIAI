#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "requests>=2.31",
# ]
# requires-python = ">=3.11"
# ///
import os
import sys
import time
import requests
import argparse
import logging

def get_unique_filename(basename):
    filename = f"{basename}.svg"
    counter = 1
    while os.path.exists(filename):
        filename = f"{basename}{counter}.svg"
        counter += 1
    return filename

def download_image(url, filename):
    response = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(response.content)

def create_prediction(prompt, token):
    url = "https://api.replicate.com/v1/models/recraft-ai/recraft-v3-svg/predictions"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Prefer": "wait"
    }
    data = {
        "input": {
            "size": "1024x1024",
            "style": "any",
            "prompt": prompt
        }
    }

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()

    return response.json()['output']

def main():
    parser = argparse.ArgumentParser(description="Generate SVG images using Recraft API.")
    parser.add_argument("prompt", type=str, help="Prompt used to generate SVG images.")
    parser.add_argument("-n", type=int, default=1, help="Number of generations to create.")
    parser.add_argument("-S", "--sleep", type=int, default=0, help="Sleep time in seconds between generations.")
    parser.add_argument("-t", "--token", type=str, default=os.getenv('REPLICATE_API_TOKEN'), help="API token for authorization.")
    parser.add_argument("-v", "--verbose", action='count', default=0, help="Increase output verbosity, e.g. -vv for more verbose.")
    parser.add_argument("--basename", type=str, default="output", help="Basename for the generated files.")

    args = parser.parse_args()

    if args.token is None:
        print("API token is not provided. Use -t or set REPLICATE_API_TOKEN environment variable.")
        sys.exit(1)

    log_level = logging.WARNING - (args.verbose * 10)
    logging.basicConfig(level=log_level)

    for i in range(args.n):
        logging.info(f"Generating image {i+1} of {args.n}...")
        try:
            image_url = create_prediction(args.prompt, args.token)
            filename = get_unique_filename(args.basename)
            download_image(image_url, filename)
            logging.info(f"Image downloaded as {filename}.")
        except requests.RequestException as e:
            logging.error(f"Error during request: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")

        if i < args.n - 1:
            logging.info(f"Sleeping for {args.sleep} seconds...")
            time.sleep(args.sleep)

if __name__ == "__main__":
    main()

# ### Explanation
# 
# 1. **Argument Parsing**:
#    - Uses `argparse` to read command-line arguments. Defaults are provided as specified.
#    - Supports verbosity with the `-v` flag.
# 
# 2. **Environment Variable**:
#    - Uses `os.getenv` to get the `REPLICATE_API_TOKEN` if not provided via the `-t` flag.
# 
# 3. **Filename Handling**:
#    - Uses a function to find a unique filename by incrementing a counter if the desired file already exists.
# 
# 4. **HTTP Request**:
#    - Uses `requests` to interact with the API. The generated SVG is downloaded directly to a file.
# 
# 5. **Logging**:
#    - Logging is used for verbosity, with higher verbosity levels providing more information during execution.
# 
# 6. **Error Handling**:
#    - Appropriate error handling for network requests and file operations.
# 
# Before running the script, ensure you have the `requests` library installed and replace or set the `REPLICATE_API_TOKEN` environment variable with your valid token:
# 
# ```bash
# pip install requests
# ```
# 
