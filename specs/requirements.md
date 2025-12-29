# CutoutAI Background Remover - Project Specifications

## Project Overview

**Name**: CutoutAI - Background Remover  
**Purpose**: Flawless background removal for t-shirt mockup preparation in Etsy workflow  
**Core Tech**: BiRefNet (BiRefNet-matting model)  
**Deployment**: Cloud-hosted (n8n/Make integration), webhook API, terminal CLI

## Current Workflow (Etsy T-Shirt Pipeline)

```
Gemini Image Gen → Slack Approval → [BACKGROUND REMOVAL] → Printify Mockup → SEO → Etsy/Shopify
                       ↓
                 Feedback loop (re-prompt if needed)
```

## Requirements

### Functional Requirements

1. **Flawless Quality**
   - NO patchy faces or artifacts
   - Clean edges on hair and fine details
   - Capture ALL elements including:
     - Bubbles
     - Small decorations
     - Floating elements
     - Text overlays

2. **Input Handling**
   - Accept various image qualities from Gemini
   - Handle non-white backgrounds (prepare for anything)
   - Process images WITH multiple small elements

3. **API/Integration**
   - Webhook endpoint for n8n
   - Base64 input/output for easy workflow integration
   - REST API for batch processing
   - Terminal CLI for manual use

4. **Cloud Deployment**
   - Host on HuggingFace Spaces or Google Cloud Run
   - Zero cold-start penalty (or minimal)
   - Handle concurrent requests

### Non-Functional Requirements

1. **Performance**
   - Sub-10 second processing for standard images
   - Batch processing capability

2. **Reliability**
   - Health check endpoint
   - Error reporting to callback URLs

## Technical Specifications

### Recommended Model Settings

```python
# BiRefNet-matting is CRITICAL for edge quality
model_variant = "matting"  # NOT "general"

# Resolution considerations
# - 1024x1024 for standard processing
# - 2048x2048 for high-res (BiRefNet_HR)

# Edge refinement is REQUIRED for mockups
edge_refinement = True
edge_radius = 2  # Subtle smoothing
```

### Known Issues to Address

1. **Artifact Prevention**
   - Downsampling large images can cause artifacts
   - Solution: Use appropriate input resolution matching model
   - Consider super-resolution post-processing if needed

2. **Multi-Element Capture**
   - BiRefNet's bilateral reference should capture small elements
   - May need to adjust detection thresholds for bubbles/decorations

3. **Edge Quality**
   - `refine_foreground` function is essential
   - Edge smoothing radius should be configurable

## API Specification

### Endpoints Required

```yaml
POST /api/v1/remove:
  input: multipart/form-data OR JSON with base64
  params:
    - model: string (matting|general|portrait|hr)
    - edge_refinement: boolean
    - edge_radius: int (1-5)
    - output_format: string (png|base64)
  output: PNG file OR JSON with base64

POST /webhook:
  input: 
    - image: file upload OR
    - image_base64: string OR
    - image_url: string
  output: JSON with base64 image

GET /health:
  output: JSON status
```

### n8n Integration

The webhook must be compatible with n8n HTTP Request node:
- Accept multipart/form-data
- Return JSON with `image_base64` field
- Support `callback_url` parameter for async notifications

## Files to Review

1. `cutoutai.py` - Core background removal logic
2. `api.py` - FastAPI server and endpoints
3. `requirements.txt` - Dependencies

## Success Criteria

- [ ] Process Gemini-generated designs without artifacts
- [ ] Capture bubbles and small decorative elements
- [ ] Clean edges suitable for Printify mockups
- [ ] Working webhook for n8n integration
- [ ] Base64 input/output for workflow compatibility
- [ ] Health check endpoint for monitoring
