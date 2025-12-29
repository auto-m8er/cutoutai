import os
import io
import base64
import numpy as np
from PIL import Image, ImageDraw
import cutoutai
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestCutoutAI")

def create_test_image(path="test_input.png"):
    """Create a synthetic test image with bubbles and a central object."""
    # 512x512 white background
    img = Image.new("RGB", (512, 512), (240, 240, 240))
    draw = ImageDraw.Draw(img)

    # Draw a "subject" (blue circle)
    draw.ellipse([150, 150, 362, 362], fill=(0, 0, 255), outline=(0, 0, 0))

    # Draw "bubbles" (small circles)
    draw.ellipse([50, 50, 80, 80], fill=(200, 200, 255, 128), outline=(100, 100, 100))
    draw.ellipse([400, 100, 430, 130], fill=(200, 200, 255, 128), outline=(100, 100, 100))
    draw.ellipse([100, 400, 140, 440], fill=(255, 200, 200, 128), outline=(100, 100, 100))

    # Draw some "fine detail" (thin lines)
    draw.line([256, 0, 256, 150], fill=(0, 0, 0), width=1)

    img.save(path)
    logger.info(f"Created test image: {path}")
    return path

def test_processing():
    """Test the core processing logic."""
    input_path = create_test_image()

    # Use 'lite' variant for faster testing if possible,
    # but the prompt asks for BiRefNet quality analysis.
    # Note: Loading the model will take time and requires internet + torch.
    # If we are in a restricted environment, this might fail.

    try:
        processor = cutoutai.CutoutAI(model_variant="lite") # Using lite for faster test

        logger.info("Running process()...")
        result = processor.process(
            input_path,
            capture_all_elements=True,
            edge_refinement=True,
            edge_radius=2,
            output_format="pil"
        )

        output_path = "test_output.png"
        result.save(output_path)
        logger.info(f"Saved result to: {output_path}")

        # Check if output is RGBA
        if result.mode == "RGBA":
            logger.info("SUCCESS: Output is in RGBA mode.")
        else:
            logger.error(f"FAILURE: Output mode is {result.mode}, expected RGBA.")

    except Exception as e:
        logger.error(f"Error during processing: {e}")
        logger.info("Note: This test requires torch and transformers to be installed and working.")

if __name__ == "__main__":
    test_processing()
