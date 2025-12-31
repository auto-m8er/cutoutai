"""
CutoutAI API Server

FastAPI server providing:
- REST API endpoints for background removal
- Webhook endpoint for n8n/Make integration
- Health check for monitoring
- Startup model preloading
"""

import io
import base64
import time
import logging
import httpx
from typing import Optional, Literal, Union
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel, Field

from cutoutai import CutoutAI, MODEL_VARIANTS, logger as cutout_logger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CutoutAI-API")

# Global model instances (by variant)
_models: dict[str, CutoutAI] = {}

def get_model(variant: str = "matting") -> CutoutAI:
    """Get or create a model instance for the specified variant."""
    global _models
    if variant not in _models:
        _models[variant] = CutoutAI(model_variant=variant)
        _models[variant].load_model()
    return _models[variant]


# Lifespan context for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: preload the default model
    print("Preloading matting model...")
    get_model("matting")
    print("Model preloaded and ready!")
    yield
    # Shutdown: cleanup
    _models.clear()


# Initialize FastAPI with lifespan
app = FastAPI(
    title="CutoutAI - Background Remover",
    description="Flawless background removal for t-shirt mockups and design workflows",
    version="1.1.0",
    lifespan=lifespan
)


# Request/Response models
class ProcessOptions(BaseModel):
    model: Literal["general", "matting", "portrait", "lite", "hr", "dynamic"] = "matting"
    capture_all_elements: bool = True
    edge_refinement: bool = True
    edge_radius: int = 2
    threshold: Optional[float] = None
    soft_threshold: bool = False
    remove_artifacts: bool = True
    min_artifact_size: int = 40
    adaptive_threshold: bool = True
    return_mask: bool = False
    output_format: Literal["png", "base64"] = "png"


class WebhookRequest(BaseModel):
    image_base64: Optional[str] = None
    image_url: Optional[str] = None
    options: Optional[ProcessOptions] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    models_loaded: list[str]
    device: str


# Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for monitoring."""
    global _models
    loaded_models = list(_models.keys())
    device = _models["matting"].device if "matting" in _models else "not loaded"
    return HealthResponse(
        status="healthy",
        version="1.1.0",
        model_loaded=len(_models) > 0,
        models_loaded=loaded_models,
        device=device
    )


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "CutoutAI - Background Remover",
        "version": "1.1.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.post("/api/v1/remove")
async def remove_bg(
    image: UploadFile = File(...),
    model: str = Form("matting"),
    edge_refinement: bool = Form(True),
    capture_all_elements: bool = Form(True),
    threshold: Optional[float] = Form(None),
    soft_threshold: bool = Form(False),
    remove_artifacts: bool = Form(True),
    adaptive_threshold: bool = Form(True),
    return_mask: bool = Form(False),
    output_format: str = Form("png")
):
    """
    Remove background from uploaded image.

    - **image**: Image file to process
    - **model**: Model variant (matting recommended for designs)
    - **edge_refinement**: Smooth edges for cleaner cutouts
    - **capture_all_elements**: Lower threshold to capture bubbles/small elements
    - **threshold**: Override mask threshold (0.0-1.0)
    - **soft_threshold**: Use soft thresholding
    - **remove_artifacts**: Remove small isolated islands from mask
    - **adaptive_threshold**: Calculate threshold based on image confidence
    - **return_mask**: Return a JSON object with both result and mask
    - **output_format**: "png" for file download, "base64" for JSON response
    """
    start_time = time.time()

    try:
        # Validate model
        if model not in MODEL_VARIANTS:
            raise HTTPException(status_code=400, detail=f"Invalid model: {model}. Available variants: {list(MODEL_VARIANTS.keys())}")

        # Read image
        contents = await image.read()

        # Validate file size (max 10MB)
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Image too large (max 10MB)")

        # Process
        processor = get_model(model)
        result = processor.process(
            contents,
            edge_refinement=edge_refinement,
            capture_all_elements=capture_all_elements,
            threshold=threshold,
            soft_threshold=soft_threshold,
            remove_artifacts=remove_artifacts,
            adaptive_threshold=adaptive_threshold,
            return_mask=return_mask,
            output_format="bytes" if output_format == "png" and not return_mask else "base64"
        )

        processing_time = time.time() - start_time

        if return_mask:
            # result is a dict here
            return JSONResponse({
                "success": True,
                "result_base64": result["result"],
                "mask_base64": result["mask"],
                "threshold_used": round(result["threshold_used"], 4),
                "processing_time_seconds": round(processing_time, 2)
            })

        if output_format == "png":
            return Response(
                content=result,
                media_type="image/png",
                headers={
                    "Content-Disposition": f'attachment; filename="{image.filename}_cutout.png"',
                    "X-Processing-Time": f"{processing_time:.2f}s"
                }
            )
        else:
            return JSONResponse({
                "success": True,
                "image_base64": result,
                "processing_time_seconds": round(processing_time, 2)
            })

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Error processing request")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/api/v1/batch")
async def batch_remove(
    images: list[UploadFile] = File(...),
    model: str = Form("matting"),
    capture_all_elements: bool = Form(True)
):
    """Process multiple images in batch."""
    start_time = time.time()
    results = []
    processor = get_model(model)
    
    for img in images:
        contents = await img.read()
        result = processor.process(
            contents,
            capture_all_elements=capture_all_elements,
            output_format="base64"
        )
        results.append({
            "filename": img.filename,
            "image_base64": result
        })
    
    total_time = time.time() - start_time
    
    return JSONResponse({
        "success": True,
        "count": len(results),
        "results": results,
        "total_processing_time_seconds": round(total_time, 2)
    })


@app.post("/webhook")
async def webhook_handler(
    request: Request,
    image: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None),
    image_url: Optional[str] = Form(None),
    model: str = Form("matting"),
    edge_refinement: bool = Form(True),
    capture_all_elements: bool = Form(True),
    edge_radius: int = Form(2),
    threshold: Optional[float] = Form(None),
    soft_threshold: bool = Form(False),
    return_mask: bool = Form(False),
    callback_url: Optional[str] = Form(None)
):
    """
    Webhook endpoint for n8n/Make integration.

    Accepts image via:
    - File upload (image)
    - Base64 encoded string (image_base64)
    - URL to fetch (image_url)

    Returns base64 encoded result for easy workflow integration.
    """
    start_time = time.time()
    logger.info(f"Webhook request received from {request.client.host}")

    try:
        # Check if JSON body instead of form
        if request.headers.get("content-type") == "application/json":
            try:
                body = await request.json()
                image_base64 = body.get("image_base64", image_base64)
                image_url = body.get("image_url", image_url)
                model = body.get("model", model)
                edge_refinement = body.get("edge_refinement", edge_refinement)
                capture_all_elements = body.get("capture_all_elements", capture_all_elements)
                edge_radius = body.get("edge_radius", edge_radius)
                threshold = body.get("threshold", threshold)
                soft_threshold = body.get("soft_threshold", soft_threshold)
                return_mask = body.get("return_mask", return_mask)
                callback_url = body.get("callback_url", callback_url)
            except Exception as e:
                logger.warning(f"Failed to parse JSON body: {e}")

        # Validate model
        if model not in MODEL_VARIANTS:
            logger.error(f"Invalid model requested: {model}")
            return JSONResponse(
                {"success": False, "error": f"Invalid model: {model}. Available: {list(MODEL_VARIANTS.keys())}"},
                status_code=400
            )

        processor = get_model(model)

        # Get image from one of the sources
        img_data = None
        if image:
            img_data = await image.read()
            logger.info(f"Using uploaded file: {image.filename}")
        elif image_base64:
            try:
                # Handle potential header in base64
                if "," in image_base64:
                    image_base64 = image_base64.split(",")[1]
                # Clean whitespace
                image_base64 = "".join(image_base64.split())
                img_data = base64.b64decode(image_base64)
                logger.info("Using base64 image data")
            except Exception as e:
                return JSONResponse({"success": False, "error": f"Invalid base64 data: {e}"}, status_code=400)
        elif image_url:
            logger.info(f"Fetching image from URL: {image_url}")
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                try:
                    response = await client.get(image_url)
                    response.raise_for_status()
                    img_data = response.content
                except httpx.HTTPStatusError as e:
                    return JSONResponse({"success": False, "error": f"Failed to fetch image: {e.response.status_code}"}, status_code=400)
                except Exception as e:
                    return JSONResponse({"success": False, "error": f"Network error: {e}"}, status_code=500)
        else:
            return JSONResponse(
                {"success": False, "error": "No image provided. Use 'image', 'image_base64', or 'image_url'"},
                status_code=400
            )

        # Validate data
        if not img_data:
            return JSONResponse({"success": False, "error": "Empty image data"}, status_code=400)

        # Process
        result = processor.process(
            img_data,
            edge_refinement=edge_refinement,
            capture_all_elements=capture_all_elements,
            edge_radius=edge_radius,
            threshold=threshold,
            soft_threshold=soft_threshold,
            return_mask=return_mask,
            output_format="base64"
        )

        processing_time = time.time() - start_time

        if isinstance(result, dict):
            response_data = {
                "success": True,
                "image_base64": result["result"],
                "mask_base64": result["mask"],
                "model_used": model,
                "threshold_used": round(result.get("threshold_used", 0), 4),
                "processing_time_seconds": round(processing_time, 2)
            }
        else:
            response_data = {
                "success": True,
                "image_base64": result,
                "model_used": model,
                "processing_time_seconds": round(processing_time, 2)
            }

        # If callback URL provided, send result there too
        if callback_url:
            logger.info(f"Sending callback to: {callback_url}")
            async with httpx.AsyncClient(timeout=10.0) as client:
                try:
                    await client.post(callback_url, json=response_data)
                except Exception as e:
                    logger.error(f"Callback failed: {e}")
                    response_data["callback_error"] = str(e)

        return JSONResponse(response_data)

    except Exception as e:
        logger.exception("Unexpected error in webhook handler")
        return JSONResponse(
            {"success": False, "error": str(e)},
            status_code=500
        )


# CLI entry point
if __name__ == "__main__":
    import uvicorn
    import argparse
    import os

    parser = argparse.ArgumentParser(description="CutoutAI API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)), help="Port number")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
