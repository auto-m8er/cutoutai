# CutoutAI Background Remover - Ralph Development Instructions

## Project Goal
Create a flawless background removal tool for the Etsy t-shirt workflow. This tool must produce perfect cutouts suitable for Printify mockups.

## Current Workflow
```
Gemini Image Gen → Slack Approval → BACKGROUND REMOVAL → Printify Mockup → SEO → Etsy/Shopify
```

## Critical Requirements

### 1. FLAWLESS Quality (Non-Negotiable)
- NO patchy faces or artifacts
- NO edge bleeding or halos
- CLEAN edges on hair and fine details
- Must look perfect on t-shirt mockups

### 2. Multi-Element Capture
The tool MUST capture ALL design elements including:
- Main subject
- Bubbles and floating decorations
- Small text or symbols
- Scattered elements (stars, sparkles, etc.)

### 3. API Integration
Must provide:
- Webhook endpoint for n8n (POST /webhook)
- REST API (POST /api/v1/remove)
- Base64 input/output support
- Health check endpoint

## Files to Review and Improve

1. **cutoutai.py** - Core processing logic
   - Uses BiRefNet-matting model (correct choice)
   - Has edge_smooth function (may need enhancement)
   - Check if multi-element capture is working properly

2. **api.py** - FastAPI server
   - Webhook endpoint exists
   - Verify n8n compatibility
   - Add any missing error handling

3. **requirements.txt** - Dependencies
   - Verify all needed packages are listed

## Improvement Tasks

### Priority 1: Quality Enhancement
- [ ] Verify BiRefNet output quality
- [ ] Test edge refinement settings
- [ ] Add adaptive thresholding for multi-element capture
- [ ] Consider adding post-processing for artifact removal

### Priority 2: API Robustness
- [ ] Add proper error responses with details
- [ ] Add request validation
- [ ] Add timeout handling for large images
- [ ] Verify callback_url functionality

### Priority 3: Deployment Ready
- [ ] Add Dockerfile for HuggingFace Spaces
- [ ] Add startup preloading (reduce first-request latency)
- [ ] Add logging for debugging

## Success Criteria
- Process Gemini-generated images with ZERO visible artifacts
- Capture ALL design elements (test with bubble/sparkle designs)
- Return base64 that works in n8n HTTP Request node
- Health endpoint returns proper status

## Reference Documents
See specs/requirements.md for detailed technical specifications.

## Notes
- This will replace the current HuggingFace BiRefNet API in the Etsy workflow
- Priority is QUALITY over speed (mockups need to be perfect)
- Test with white AND non-white backgrounds (Gemini may vary)
