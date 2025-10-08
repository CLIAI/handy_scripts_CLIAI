#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "Pillow>=10.0.0",
#   "opencv-python>=4.8.0",
#   "pytesseract>=0.3.10",
#   "numpy>=1.24.0",
# ]
# requires-python = ">=3.11"
# ///

# OCR with Tesseract including enhancing options leverages adaptive and
# simple thresholding to boost text extraction accuracy; employs
# preprocessing for grayscale conversion and bounding box details as
# differentiators; core features include user-configurable options for
# grayscale, thresholding, OCR engine, page segmentation, and language
# support; with additional functionalities like auto-preprocessing and
# bounding box extraction, this tool provides detailed text recognition
# data while ensuring ease of use through command-line interface.
#
# published:
# * https://github.com/CLIAI/handy_scripts/blob/main/ocr_tesseract.py
# * https://gist.github.com/gwpl/aa26593b575c93c76178a1bcc2afa9eb
#
# Keep sources:
# * https://x.com/i/grok/share/mbO9qX38FWwWq1CvRMvG4rpeJ
# * https://chatgpt.com/share/67c9e100-84d4-8007-b9ef-c44c419e7e13
# * https://www.phind.com/search/cm7xn42px00002v6s3gh7xg8e


import sys
import argparse
import json
import os
from PIL import Image
import subprocess  # Added to handle the --view parameter

# Try to import required libraries with helpful error messages if they fail
try:
    import cv2
except ImportError:
    print("ERROR: import cv2 failed")
    print("For ArchLinux users, you may want to install:")
    print("# sudo pacman -S python-opencv opencv-samples opencv")
    print("For other distributions, use your package manager or pip:")
    print("# pip install opencv-python")
    sys.exit(1)

try:
    import pytesseract
except ImportError:
    print("ERROR: import pytesseract failed")
    print("For ArchLinux users, you may want to install:")
    print("# sudo pacman -S python-pytesseract python-pyocr")
    print("For other distributions, use your package manager or pip:")
    print("# pip install pytesseract")
    sys.exit(1)

def preprocess_image(image, grayscale=True, threshold=False, threshold_value=150, adaptive=False):
    """Preprocess the image with specified options."""
    processed = image.copy()
    if grayscale:
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    if threshold:
        if adaptive:
            processed = cv2.adaptiveThreshold(processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        else:
            _, processed = cv2.threshold(processed, threshold_value, 255, cv2.THRESH_BINARY)
    return processed

def draw_bounding_boxes(image, data, output_path):
    """Draw bounding boxes on the image and save it."""
    try:
        import numpy as np
        # Make a copy of the image to draw on
        img_with_boxes = image.copy()
        
        # Draw rectangles for each detected text area
        for i in range(len(data["text"])):
            if data["text"][i].strip():
                x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Add text label with confidence
                text = f"{data['text'][i]} ({data['conf'][i]}%)"
                cv2.putText(img_with_boxes, text, (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Save the image with bounding boxes
        cv2.imwrite(output_path, img_with_boxes)
        return True
    except ImportError as e:
        print(f"ERROR: Cannot generate image with bounding boxes due to missing import: {e}", file=sys.stderr)
        return False

def perform_ocr(image_path, preprocess_options, tesseract_config, return_bounding_boxes=False, draw_boxes_path=None):
    """Run OCR on the image with the given settings."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to load image at {image_path}")
    processed_image = preprocess_image(image, **preprocess_options)
    pil_image = Image.fromarray(processed_image)
    extracted_text = pytesseract.image_to_string(pil_image, config=tesseract_config)
    output = {"text": extracted_text}
    
    # Always get bounding box data if we need to draw boxes
    if return_bounding_boxes or draw_boxes_path:
        data = pytesseract.image_to_data(pil_image, config=tesseract_config, output_type=pytesseract.Output.DICT)
        output["data"] = data
        
        # Draw bounding boxes if requested
        if draw_boxes_path:
            draw_bounding_boxes(image, data, draw_boxes_path)
    
    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tesseract OCR with Preprocessing Options")
    parser.add_argument("image_path", type=str, help="Path to the image file")
    parser.add_argument("--grayscale", "-g", action="store_true", default=True, help="Convert image to grayscale")
    parser.add_argument("--no-grayscale", action="store_false", dest="grayscale", help="Don't convert image to grayscale")
    parser.add_argument("--threshold", "-th", action="store_true", default=True, help="Apply simple thresholding")
    parser.add_argument("--no-threshold", action="store_false", dest="threshold", help="Don't apply thresholding")
    parser.add_argument("--threshold_value", "--tv", type=int, default=150, help="Threshold value for simple thresholding")
    parser.add_argument("--adaptive", "-a", action="store_true", default=False, help="Use adaptive thresholding")
    parser.add_argument("--psm", type=int, default=3, help="Page segmentation mode for Tesseract")
    parser.add_argument("--oem", type=int, default=3, help="OCR Engine mode for Tesseract")
    parser.add_argument("--language", "--lang", "-l", type=str, default="eng", help="Language for OCR")
    parser.add_argument("--bounding-boxes", "-B", action="store_true", help="Return bounding box information")
    parser.add_argument("--draw-bounding-boxes", "--dbb", metavar="FILENAME", 
                        help="Draw bounding boxes on the image and save to specified file. Use 'auto' or '-' to automatically name the file as [input_image].bb.png")
    parser.add_argument("--jsonl", "-j", action="store_true", help="Output results in JSONL format (one JSON object per line)")
    parser.add_argument("--auto_preprocess", "--pre", action="store_true", help="Automatically use adaptive thresholding")
    parser.add_argument("--view", "-V", type=str,
                        help="Command line to view bounding box image after generation. If '{}' is present, the bounding box "
                             "image filename will replace '{}'. Otherwise the bounding box image will be appended as the final argument.")
    args = parser.parse_args()

    tesseract_config = f"--psm {args.psm} --oem {args.oem} -l {args.language}"
    preprocess_options = {
        "grayscale": args.grayscale,
        "threshold": args.threshold,
        "threshold_value": args.threshold_value,
        "adaptive": args.adaptive or args.auto_preprocess
    }
    
    # If drawing bounding boxes is requested, ensure bounding_boxes is also enabled
    return_bounding_boxes = args.bounding_boxes or (args.draw_bounding_boxes is not None)
    
    # Ensure the output path for drawn bounding boxes has .png extension
    draw_boxes_path = None
    if args.draw_bounding_boxes:
        # Handle special cases for auto-naming
        if args.draw_bounding_boxes == '-' or args.draw_bounding_boxes.lower() == 'auto':
            # Generate filename based on input image name
            base_name = os.path.splitext(args.image_path)[0]
            draw_boxes_path = f"{base_name}.bb.png"
        else:
            draw_boxes_path = args.draw_bounding_boxes
            if not draw_boxes_path.lower().endswith('.png'):
                draw_boxes_path += '.png'

    result = perform_ocr(args.image_path, preprocess_options, tesseract_config, 
                         return_bounding_boxes, draw_boxes_path)
    
    if args.jsonl:
        # Output full text as first JSONL entry
        print(json.dumps({"type": "full_text", "text": result["text"]}))
        
        # Output each text box as a separate JSONL entry if bounding boxes requested
        if args.bounding_boxes:
            for i in range(len(result["data"]["text"])):
                if result["data"]["text"][i].strip():
                    box_data = {
                        "type": "text_box",
                        "text": result["data"]["text"][i],
                        "x": result["data"]["left"][i],
                        "y": result["data"]["top"][i],
                        "w": result["data"]["width"][i],
                        "h": result["data"]["height"][i],
                        "confidence": result["data"]["conf"][i]
                    }
                    print(json.dumps(box_data))
    else:
        # Original human-readable output format
        print("Extracted Text:", result["text"])
        if args.bounding_boxes and "data" in result:
            print("\nBounding Box Data:")
            for i in range(len(result["data"]["text"])):
                if result["data"]["text"][i].strip():
                    print(f"Text: '{result['data']['text'][i]}' | Box: [{result['data']['left'][i]}, {result['data']['top'][i]}, {result['data']['width'][i]}, {result['data']['height'][i]}] | Confidence: {result['data']['conf'][i]}")
    
    # If we generated a bounding box image and the user wants to view it, run the specified command
    if draw_boxes_path and args.view:
        if '{}' in args.view:
            view_cmd = args.view.format(draw_boxes_path)
            view_cmd_list = view_cmd.split()
        else:
            view_cmd_list = args.view.split() + [draw_boxes_path]
        
        subprocess.run(view_cmd_list)
