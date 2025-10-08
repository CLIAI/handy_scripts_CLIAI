#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "requests>=2.31",
#   "replicate>=0.25",
# ]
# requires-python = ">=3.11"
# ///

import argparse
import os
import sys
import json
import requests
import replicate

def available_ocr_models():
    return ["cuuupid-marker", "cudanexus-nougat", "awilliamson10-meta-nougat"]

def generate_ocr(input_file, output_file, model):
    if model == "cuuupid-marker":
        output = replicate.run(
            "cuuupid/marker:9c67051309f6d10ca139489f15fcb5ebc4866a3734af537c181fb13bc719d280",
            input={
                "dpi": 400,
                "lang": "English",
                "document": open(input_file, "rb"),
                "enable_editor": False,
                "parallel_factor": 10
            }
        )
    elif model == "cudanexus-nougat":
        output = replicate.run(
            "cudanexus/nougat:d0b4e90da423598ff84debc9115bf891dd819843600ad842c0c178e3571f9e76",
            input={"pdf_file": open(input_file, "rb")}
        )
    elif model == "awilliamson10-meta-nougat":
        output = replicate.run(
            "awilliamson10/meta-nougat:872fa99400b0eeb8bfc82ef433aa378976b4311178ff64fed439470249902071",
            input={"pdf_link": input_file}
        )
    else:
        raise ValueError(f"Unsupported model: {model}")

    json_output_file = output_file.rsplit('.', 1)[0] + '.json'
    with open(json_output_file, "w") as f:
        json.dump(output, f)

    markdown_url = output.get('markdown')
    if markdown_url:
        response = requests.get(markdown_url)
        response.raise_for_status()
        with open(output_file, "w") as f:
            f.write(response.text)
    else:
        print("No markdown URL found in the server response.")

def main():
    parser = argparse.ArgumentParser(description="OCR PDF to Markdown converter")
    parser.add_argument("input_file", help="Input PDF file")
    parser.add_argument("-o", "--output", help="Custom output filename")
    parser.add_argument("-m", "--model", choices=available_ocr_models(), default="marker", help="OCR model to use")
    parser.add_argument("--all", action='store_true', help="Run processing through all available OCR models")
    parser.add_argument("-D", "--output-dir", help="Output directory")
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist.")
        sys.exit(1)

    output_dir = args.output_dir if args.output_dir else ""
    if output_dir and not os.path.isdir(output_dir):
        print(f"Error: Output directory '{output_dir}' does not exist or is not a directory.")
        sys.exit(1)

    output_file = os.path.join(output_dir, args.output if args.output else os.path.splitext(os.path.basename(args.input_file))[0] + ".md")
    json_output_file = output_file.rsplit('.', 1)[0] + '.md.json'

    if os.path.exists(output_file) or os.path.exists(json_output_file):
        response = input(f"Warning: Output file '{output_file}' or '{json_output_file}' already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Operation cancelled.")
            sys.exit(0)

    if os.path.exists(output_file):
        response = input(f"Warning: Output file '{output_file}' already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Operation cancelled.")
            sys.exit(0)

    if args.all:
        for model in available_ocr_models():
            model_output_file = os.path.join(output_dir, output_file.rsplit('.', 1)[0] + f'.{model}.md')
            model_json_output_file = model_output_file.rsplit('.', 1)[0] + '.md.json'
            if os.path.exists(model_output_file) or os.path.exists(model_json_output_file):
                print(f"Warning: Output file '{model_output_file}' or '{model_json_output_file}' already exists. Skipping model {model}.")
                continue
            generate_ocr(args.input_file, model_output_file, model)
    else:
        generate_ocr(args.input_file, output_file, args.model)

if __name__ == "__main__":
    main()
