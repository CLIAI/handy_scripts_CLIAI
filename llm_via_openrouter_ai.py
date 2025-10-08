#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "requests>=2.31",
# ]
# requires-python = ">=3.11"
# ///
import argparse
import configparser
import json
import os
import sys

import requests

def check_api_key() -> str:
    api_key = os.getenv('OPENROUTER_API_KEY')
    if api_key is None:
        raise ValueError("OPENROUTER_API_KEY environment variable was not found")
    return api_key

OPENROUTER_API_KEY = check_api_key()

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='OpenRouter API script.')
    parser.add_argument('-F', '--file', help='File to use as content')
    parser.add_argument('-M', '--ask-model', action='store_true', help='Ask which model to use from config file')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress progress messages')
    parser.add_argument('-m', '--model', help='Model name or number to use. You can provide model by name or number from the list. You can provide multiple separated by comma to query multiple models.')
    parser.add_argument('-j', '--json', action='store_true', help='Return the full JSON response')
    parser.add_argument('--json-stderr', action='store_true', help='Output content to stdout and full JSON to stderr')
    parser.add_argument('--message', '--msg', help='Specify a single message to send the LLM, process reply then exit')
    parser.add_argument('--message-file', '-f', help='Specify a file containing the message to send the LLM, process reply, then exit')
    return parser.parse_args()

args = parse_arguments()

def log_message(message: str, quiet: bool) -> None:
    if not quiet:
        print(f"> {message}", file=sys.stderr)

# Config file path
config_file_path = os.path.expanduser('~/.config/openrouter_models.conf')

# Example config content
example_config_content = """
[models]
lumimaid = neversleep/llama-3-lumimaid-70b
euryale = sao10k/l3-euryale-70b
dolphin22b = cognitivecomputations/dolphin-mixtral-8x22b
noromaid20b = neversleep/noromaid-20b
toppy7b = undi95/toppy-m-7b
"""
def read_config(config_file_path: str, example_config_content: str) -> configparser.ConfigParser:
    if not os.path.isfile(config_file_path) and args.quiet:
        sys.exit(1)
    if not os.path.isfile(config_file_path):
        log_message(f"Config file not found at {config_file_path}", args.quiet)
        log_message("Example config file content:", args.quiet)
        log_message(example_config_content, args.quiet)
        create_config = input("Would you like to create this example config file? (yes/Y to confirm): ").strip().lower()
        if create_config in ['yes', 'y']:
            os.makedirs(os.path.dirname(config_file_path), exist_ok=True)
            with open(config_file_path, 'w') as config_file:
                config_file.write(example_config_content)
            log_message(f"Config file created at {config_file_path}", args.quiet)
        else:
            sys.exit(1)

    config = configparser.ConfigParser()
    config.read(config_file_path)
    return config

config = read_config(config_file_path, example_config_content)
models = config['models']
default_model = models.get('default', next(iter(models.values())))

def determine_models(args: argparse.Namespace, models: configparser.SectionProxy, default_model: str, config_file_path: str) -> list:
    selected_models = []
    if args.model:
        for model in args.model.split(','):
            if model.isdigit():
                model_list = list(models.keys())
                selected_models.append(models[model_list[int(model) - 1]])
            else:
                selected_models.append(model.strip())
    elif args.ask_model:
        print(f"Available models (from {config_file_path}):")
        model_list = list(models.keys())
        for i, model in enumerate(model_list):
            print(f"{i + 1}. {models[model]}")
        choices = input("Select models providing numbers or names spearated by comma: ").split(',')
        for choice in choices:
            if choice.isdigit():
                selected_models.append(models[model_list[int(choice) - 1]])
            else:
                selected_models.append(choice.strip())
    else:
        selected_models.append(default_model)
    return selected_models

selected_models = determine_models(args, models, default_model, config_file_path)
log_message(f"Selected models: {', '.join(selected_models)}", args.quiet)

def read_content(args: argparse.Namespace) -> str:
    if args.message:
        return args.message
    elif args.message_file:
        if not os.path.isfile(args.message_file):
            raise FileNotFoundError(f"File {args.message_file} does not exist")
        with open(args.message_file, 'r') as file:
            return file.read()
    elif args.file:
        if not os.path.isfile(args.file):
            raise FileNotFoundError(f"File {args.file} does not exist")
        with open(args.file, 'r') as file:
            return file.read()
    else:
        log_message("Reading until End of File (press Ctrl-D after last line):", args.quiet)
        return sys.stdin.read()

args = parse_arguments()

def log_message(message: str, quiet: bool) -> None:
    if not quiet:
        print(f"> {message}", file=sys.stderr)

args = parse_arguments()

def log_message(message: str, quiet: bool) -> None:
    if not quiet:
        print(f"> {message}", file=sys.stderr)

content = read_content(args)
log_message(f"Prompt: {content}", args.quiet)


def send_request_to_server(selected_model: str, content: str) -> dict:
    log_message(f"Sending to the server...", args.quiet)
    log_message(f"MODEL: {selected_model}", args.quiet)
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        },
        data=json.dumps({
            "model": selected_model,
            "messages": [
                { "role": "user", "content": content }
            ]
        })
    )
    log_message("This is the response that I got", args.quiet)
    return response.json()

def print_content(response_json: dict) -> None:
    if 'choices' in response_json and len(response_json['choices']) > 0:
        print(response_json['choices'][0]['message']['content'])
    else:
        print("null")

def process_response(response_json: dict) -> None:
    if args.json:
        print(json.dumps(response_json, indent=4))
    elif args.json_stderr:
        print_content(response_json)
        log_message(json.dumps(response_json, indent=4), args.quiet)
    else:
        print_content(response_json)

for selected_model in selected_models:
    for selected_model in selected_models:
        response_json = send_request_to_server(selected_model, content)
        process_response(response_json)
