#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "requests>=2.31",
#   "openai>=1.0",
#   "ell-ai>=0.0.13",
# ]
# requires-python = ">=3.11"
# ///

# Important Note to Keep:
# This script imports several libraries in try blocks by design.
# This allows the script to failover to other implementations if certain dependencies are not present in the system.

import os
import sys
import json
import requests
import argparse
import inspect

# Global variable
verbose_flag = False

def INFO(message):
    print(message, file=sys.stderr)

def DEBUG(message):
    if verbose_flag:
        print(message, file=sys.stderr)

def DEBUG_function_name(llmreq):
    """
    Handy function to print function name when being called in debug mode.
    """
    DEBUG("FUNCTION:{fx_name}({llmreq})".format(fx_name=inspect.currentframe().f_back.f_code.co_name, llmreq=llmreq.to_json()))

class LLMRequest:
    def __init__(self):
        self.SYSTEM_MESSAGE = "You are a helpful AI assistant."
        self.PROMPT = "Please provide a information dense (maximise information per words) answer to advanced sophisticated reader the following question: Describe answer `42` to universe."
        self.MODEL = "gpt-4o"
        self.model = self.MODEL
        self.TEMPERATURE = 0.7
        self.api_keys = {}
        self.api_keys["openai"] = os.environ.get("OPENAI_API_KEY")
        self.prompt = None

    def to_json_with_private_key(self):
        return json.dumps(self.__dict__)
    
    def to_json(self):
        # make a copy of the object
        llmreq_copy = self.__dict__.copy()
        # remove api_keys from the copy
        llmreq_copy.pop('api_keys', None)
        # serialize the copy to JSON
        return json.dumps(llmreq_copy)

    @classmethod
    def from_json(cls, json_str):
        llmreq = cls()
        llmreq.__dict__.update(json.loads(json_str))
        return llmreq

llmreq = LLMRequest()

if not llmreq.api_keys["openai"]:
    print("Error: OPENAI_API_KEY environment variable not set.")
    exit(1)

def get_ell_response(req):
    prompt = req.PROMPT
    model = req.MODEL
    DEBUG_function_name(req)
    try:
        import ell
        @ell.simple(model=model)
        def ell_response(prompt: str):
            """{req.SYSTEM_MESSAGE}"""
            return prompt
        return ell_response(prompt)
    except ImportError:
        return None

def get_openai_response(req):
    DEBUG_function_name(req)
    try:
        import openai
        openai.api_key = req.api_keys["openai"]
        response = openai.chat.completions.create(
            model=req.model,
            messages=[
                {"role": "system", "content": req.SYSTEM_MESSAGE},
                {"role": "user", "content": req.prompt}
            ],
            temperature=req.TEMPERATURE
        )
        return response.choices[0].message.content
    except ImportError:
        return None

def get_rest_response(req):
    DEBUG_function_name(req)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {req.api_keys["openai"]}"
    }

    data = {
        "model": req.model,
        "messages": [
            {"role": "system", "content": req.SYSTEM_MESSAGE},
            {"role": "user", "content": req.prompt}
        ],
        "temperature": req.TEMPERATURE
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error: {response.status_code} - {response.text}"

def get_response(req, args):
    if args.implementation == 'ell':
        return get_ell_response(req)
    elif args.implementation == 'openai_library':
        return get_openai_response(req)
    elif args.implementation == 'openai_rest':
        return get_rest_response(req)
    else:
        DEBUG("No implementation specified. Trying all...")
        response = get_ell_response(req)
        if response is not None:
            return response
        response = get_openai_response(req)
        if response is not None:
            return response
        response = get_rest_response(req)
    return response

def parse_args(llmreq):
    parser = argparse.ArgumentParser(description='Process user input with failover between ELL, OpenAI, and REST.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Display verbose output.')
    parser.add_argument('-s', '--sysmsg', type=str, help='Set system message.')
    parser.add_argument('-S', '--sysmsgfile', type=str, help='Set system message file.')
    parser.add_argument('-p', '--prompt', type=str, help='Set prompt.')
    parser.add_argument('-P', '--promptfile', type=str, help='Set prompt file.')
    parser.add_argument('-i', '--implementation', type=str, choices=['ell', 'openai_library', 'openai_rest'], help='Enforce using one of the available implementations.')
    parser.add_argument('-t', '--temperature', type=float, help='Set temperature.')
    parser.add_argument('-m', '--model', type=str, help='Set model.')
    parser.add_argument('-d', '--dump', action='store_true', help='Dump the current llmrequration to stdout as a JSON string.')
    parser.add_argument('-l', '--load', type=str, help='Load the llmrequration from a JSON string.')
    args = parser.parse_args()
    if args.dump:
        print(llmreq.to_json())
        sys.exit(0)
    if args.load:
        llmreq = LLMRequest.from_json(args.load)
    return args, llmreq

def main():
    llmreq = LLMRequest()
    args, llmreq = parse_args(llmreq)
    model = args.model if args.model else llmreq.MODEL
    if args.verbose:
        global verbose_flag
        verbose_flag=True
    llmreq.TEMPERATURE = args.temperature if args.temperature else llmreq.TEMPERATURE
    if args.promptfile:
        if args.promptfile == '-':
            llmreq.PROMPT = sys.stdin.read().strip()
        else:
            with open(args.promptfile, 'r') as f:
                llmreq.PROMPT = f.read().strip()
    else:
        llmreq.PROMPT = args.prompt if args.prompt else sys.stdin.read().strip()

    if args.sysmsgfile:
        with open(args.sysmsgfile, 'r') as f:
            llmreq.SYSTEM_MESSAGE = f.read().strip()
    else:
        llmreq.SYSTEM_MESSAGE = args.sysmsg if args.sysmsg else llmreq.SYSTEM_MESSAGE

    response = get_response(llmreq, args)
    if response:
        DEBUG("RESPONSE:")
        print(response)
    else:
        print("Failed to get a response from the AI.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main ()
