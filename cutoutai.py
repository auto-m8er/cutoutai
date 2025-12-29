"""
CutoutAI - Enhanced Background Removal for Perfect T-Shirt Mockups

Built on BiRefNet for flawless background removal with:
- Multi-element capture (bubbles, decorations, small details)
- Edge refinement for clean cutouts
- Optimized for Printify mockup preparation
"""

import io
import base64
import time
import logging
from typing import Optional, Literal, Union
from pathlib import Path

import torch
import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CutoutAI")

# Model variants available
MODEL_VARIANTS = {
    "general": "ZhengPeng7/BiRefNet",           # General use
    "matting": "ZhengPeng7/BiRefNet-matting",   # Best for complex edges
    "portrait": "ZhengPeng7/BiRefNet-portrait", # Faces/people
    "lite": "ZhengPeng7/BiRefNet_lite",         # Faster, smaller
    "hr": "ZhengPeng7/BiRefNet_HR",             # High resolution (2K)
    "dynamic": "ZhengPeng7/BiRefNet_dynamic",   # Variable resolution
}

# Default image transforms
def get_transforms(size: int = 1024):
    """Get preprocessing transforms for BiRefNet."""
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def refine_foreground(image: Image.Image, mask: Image.Image) -> Image.Image:
    """
    Apply mask to image with refined edges for flawless cutouts.
    
    This is critical for t-shirt mockups - ensures:
    - No patchy faces or artifacts
    - Clean edges on hair and fine details
    - All small elements (bubbles, decorations) captured
    """
    # Convert to RGBA
    image = image.convert("RGBA")
    mask = mask.convert("L")
    
    # Resize mask to match image if needed
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.LANCZOS)
    
    # Apply mask as alpha channel
    result = Image.new("RGBA", image.size, (0, 0, 0, 0))
    result.paste(image, mask=mask)
    
    return result


def edge_smooth(mask: Image.Image, radius: int = 2, preserve_details: bool = True) -> Image.Image:
    """
    Apply edge smoothing while preserving fine details.

    Args:
        mask: Binary or grayscale mask
        radius: Smoothing intensity (1-5 recommended)
        preserve_details: If True, use morphological ops instead of blur
    """
    if radius <= 0:
        return mask

    if preserve_details:
        # Use morphological operations to clean edges without losing detail
        # Erosion removes thin protrusions (noise)
        # size must be odd
        size = 2 * radius + 1
        eroded = mask.filter(ImageFilter.MinFilter(size))
        # Dilation restores the shape
        smoothed = eroded.filter(ImageFilter.MaxFilter(size))

        # Optional: slight median filter to remove salt-and-pepper noise
        if radius > 1:
            smoothed = smoothed.filter(ImageFilter.MedianFilter(3))
    else:
        # Fall back to gaussian blur for softer edges
        smoothed = mask.filter(ImageFilter.GaussianBlur(radius=radius))

    return smoothed


def remove_small_artifacts(mask: Image.Image, min_size: int = 100) -> Image.Image:
    """
    Remove small isolated 'islands' from the mask that are likely artifacts.

    Args:
        mask: Grayscale mask (PIL Image)
        min_size: Minimum pixel area to keep
    """
    import numpy as np
    from scipy import ndimage

    # Convert to binary
    mask_np = np.array(mask) > 128

    # Label connected components
    label_im, nb_labels = ndimage.label(mask_np)

    # Calculate sizes of components
    sizes = ndimage.sum(mask_np, label_im, range(nb_labels + 1))

    # Identify components that are too small
    mask_size = sizes < min_size
    remove_pixel = mask_size[label_im]

    # Remove small components
    mask_np[remove_pixel] = 0

    return Image.fromarray((mask_np * 255).astype(np.uint8))


def calculate_adaptive_threshold(pred: np.ndarray, base_threshold: float = 0.2) -> float:
    """
    Calculate an adaptive threshold based on the prediction distribution.
    Useful for capturing small design elements without introducing too much noise.
    """
    # Simple adaptive approach: if there are many low-confidence pixels,
    # we might be looking at a design with many small elements (bubbles, etc.)
    # We can use a percentile-based approach or Otsu's method if appropriate

    # For now, let's use a simple heuristic:
    # If the 95th percentile is low, it's a very faint design, lower the threshold further
    p95 = np.percentile(pred, 95)
    if p95 < 0.5:
        return max(0.05, base_threshold * 0.5)

    return base_threshold


def apply_threshold(pred: np.ndarray, threshold: float = 0.4, soft: bool = False) -> np.ndarray:
    """
    Apply threshold to mask for cleaner binary edges.

    Args:
        pred: Prediction array (0-1 range)
        threshold: Cutoff value (pixels below become 0, above become 1)
        soft: If True, use a soft threshold (keep low confidence regions as semi-transparent)

    Returns:
        Thresholded array
    """
    if soft:
        # Sigmoid-like soft thresholding
        # Regions near threshold are preserved but dimmed
        # Steepness of 15 provides a good balance between sharp and soft
        return 1.0 / (1.0 + np.exp(-15 * (pred - threshold)))

    return np.where(pred > threshold, 1.0, 0.0)


class CutoutAI:
    """
    Enhanced background removal optimized for t-shirt mockup preparation.
    
    Key features:
    - Captures ALL elements including bubbles, small decorations
    - Flawless edge quality with no artifacts
    - Multiple model options for different use cases
    """
    
    def __init__(
        self,
        model_variant: Literal["general", "matting", "portrait", "lite", "hr", "dynamic"] = "matting",
        device: Optional[str] = None,
        resolution: int = 1024
    ):
        """
        Initialize CutoutAI.
        
        Args:
            model_variant: Which BiRefNet model to use
                - "matting": Best for complex edges, hair, fine details (RECOMMENDED)
                - "general": Standard background removal
                - "portrait": Optimized for faces/people
                - "lite": Faster processing, lower quality
                - "hr": High resolution up to 2K
                - "dynamic": Variable resolution support
            device: "cuda", "cpu", or None for auto-detect
            resolution: Processing resolution (1024 or 2048 for hr model)
        """
        self.model_variant = model_variant
        self.model_name = MODEL_VARIANTS[model_variant]
        self.resolution = resolution
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model = None
        self.transforms = get_transforms(resolution)
    
    def load_model(self):
        """Load the BiRefNet model from HuggingFace."""
        if self.model is not None:
            return
        
        from transformers import AutoModelForImageSegmentation
        
        print(f"Loading {self.model_name}...")
        self.model = AutoModelForImageSegmentation.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}")
    
    def process(
        self,
        image: Union[str, Path, Image.Image, bytes],
        capture_all_elements: bool = True,
        edge_refinement: bool = True,
        edge_radius: int = 2,
        threshold: Optional[float] = None,
        soft_threshold: bool = False,
        preserve_details: bool = True,
        remove_artifacts: bool = True,
        min_artifact_size: int = 40,
        adaptive_threshold: bool = True,
        return_mask: bool = False,
        output_format: Literal["pil", "bytes", "base64"] = "pil"
    ) -> Union[Image.Image, bytes, str, dict]:
        """
        Remove background from image with enhanced quality.

        Args:
            image: Input image (path, PIL Image, or bytes)
            capture_all_elements: Use lower threshold to capture bubbles/small elements
            edge_refinement: Apply edge smoothing for cleaner cutouts
            edge_radius: Smoothing intensity (1-5, default 2)
            threshold: Override mask threshold (0.0-1.0, None for auto)
            soft_threshold: Use soft thresholding for smoother transitions
            preserve_details: Use morphological ops instead of blur
            remove_artifacts: Remove small isolated islands from mask
            min_artifact_size: Minimum pixel area for islands to keep
            adaptive_threshold: Calculate threshold based on image confidence
            return_mask: If True, return a dict containing both result and mask
            output_format: Return format ("pil", "bytes", "base64")

        Returns:
            Processed image with transparent background (or dict if return_mask=True)
        """
        start_time = time.time()
        logger.info(f"Processing image with variant: {self.model_variant}")
        self.load_model()

        # Load image
        try:
            if isinstance(image, (str, Path)):
                pil_image = Image.open(image).convert("RGB")
            elif isinstance(image, bytes):
                pil_image = Image.open(io.BytesIO(image)).convert("RGB")
            else:
                pil_image = image.convert("RGB")
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            raise ValueError(f"Invalid image input: {e}")

        original_size = pil_image.size
        logger.info(f"Image size: {original_size}")

        # Preprocess
        input_tensor = self.transforms(pil_image).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(input_tensor)

        # Get prediction mask
        if isinstance(outputs, (list, tuple)):
            pred = outputs[0]
        else:
            pred = outputs

        # Convert to numpy
        pred = pred.squeeze().cpu().numpy()

        # Apply thresholding for cleaner edges
        # Lower threshold captures more (bubbles, small elements)
        # Higher threshold is more selective
        if threshold is not None:
            mask_threshold = threshold
        elif capture_all_elements:
            mask_threshold = 0.2  # Base low threshold
            if adaptive_threshold:
                mask_threshold = calculate_adaptive_threshold(pred, mask_threshold)
        else:
            mask_threshold = 0.4  # Standard threshold

        logger.info(f"Using threshold: {mask_threshold:.4f} (soft: {soft_threshold})")
        pred = apply_threshold(pred, mask_threshold, soft=soft_threshold)

        # Convert to PIL mask
        pred = (pred * 255).astype(np.uint8)
        mask = Image.fromarray(pred).resize(original_size, Image.LANCZOS)

        # Remove small artifacts if requested
        if remove_artifacts:
            logger.info(f"Removing small artifacts (min_size: {min_artifact_size})")
            try:
                mask = remove_small_artifacts(mask, min_size=min_artifact_size)
            except ImportError:
                logger.warning("Scipy not installed, skipping artifact removal")

        # Edge refinement for cleaner cutouts
        if edge_refinement:
            logger.info(f"Applying edge refinement (radius: {edge_radius})")
            mask = edge_smooth(mask, radius=edge_radius, preserve_details=preserve_details)

        # Apply mask to get final result
        result = refine_foreground(pil_image, mask)

        # Record processing time
        self._last_processing_time = time.time() - start_time
        logger.info(f"Processing completed in {self._last_processing_time:.2f}s")

        # Prepare outputs
        if return_mask:
            return {
                "result": self._format_output(result, output_format),
                "mask": self._format_output(mask, output_format),
                "threshold_used": mask_threshold,
                "processing_time": self._last_processing_time
            }

        return self._format_output(result, output_format)

    def _format_output(self, image: Image.Image, output_format: str) -> Union[Image.Image, bytes, str]:
        """Format PIL Image to requested output format."""
        if output_format == "pil":
            return image
        elif output_format == "bytes":
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            return buffer.getvalue()
        elif output_format == "base64":
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode()
        return image

    @property
    def last_processing_time(self) -> float:
        """Get the processing time of the last operation in seconds."""
        return getattr(self, '_last_processing_time', 0.0)
    
    def process_batch(
        self,
        images: list,
        **kwargs
    ) -> list:
        """Process multiple images."""
        return [self.process(img, **kwargs) for img in images]


# Convenience function
def remove_background(
    image: Union[str, Path, Image.Image, bytes],
    model: str = "matting",
    capture_all_elements: bool = True,
    edge_refinement: bool = True,
    **kwargs
) -> Image.Image:
    """
    Quick function to remove background from an image.
    
    Args:
        image: Input image
        model: Model variant ("matting" recommended for t-shirt designs)
        capture_all_elements: Capture bubbles, small elements (uses lower threshold)
        edge_refinement: Smooth edges for clean mockups
    
    Returns:
        PIL Image with transparent background
    """
    processor = CutoutAI(model_variant=model)
    return processor.process(
        image,
        capture_all_elements=capture_all_elements,
        edge_refinement=edge_refinement,
        **kwargs
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CutoutAI Background Remover")
    parser.add_argument("input", help="Input image path")
    parser.add_argument("-o", "--output", help="Output path", default=None)
    parser.add_argument("-m", "--model", choices=list(MODEL_VARIANTS.keys()), 
                       default="matting", help="Model variant")
    parser.add_argument("--no-edge-refinement", action="store_true",
                       help="Disable edge refinement")
    parser.add_argument("--threshold", type=float, default=None,
                       help="Mask threshold (0.0-1.0)")
    parser.add_argument("--capture-all", action="store_true", default=True,
                       help="Use lower threshold to capture small elements")
    
    args = parser.parse_args()
    
    # Process
    result = remove_background(
        args.input,
        model=args.model,
        edge_refinement=not args.no_edge_refinement,
        capture_all_elements=args.capture_all,
        threshold=args.threshold
    )
    
    # Save
    output_path = args.output or args.input.rsplit(".", 1)[0] + "_cutout.png"
    result.save(output_path)
    print(f"Saved to: {output_path}")
