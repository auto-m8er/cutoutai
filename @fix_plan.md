# CutoutAI - Task Priority List

## Completed
- [x] Create basic cutoutai.py with BiRefNet integration
- [x] Create api.py with webhook endpoint
- [x] Add edge smoothing function
- [x] Add requirements.txt
- [x] Create project documentation
- [x] Add mask thresholding (0.2 for capture_all, 0.4 standard)
- [x] Implement capture_all_elements with lower threshold
- [x] Replace blur with morphological edge processing (preserves details)
- [x] Add startup model preloading
- [x] Add processing time to responses
- [x] Use model parameter in webhook
- [x] Add input validation (10MB limit)
- [x] Add Dockerfile for HuggingFace Spaces deployment
- [x] Add processing time logging
- [x] Add optional debug mode with intermediate outputs (return_mask=True)
- [x] Add artifact removal (scipy ndimage)
- [x] Add adaptive thresholding

## In Progress
- [ ] Test with Gemini-generated images

## High Priority
- [ ] Add Dockerfile for HuggingFace Spaces deployment
- [ ] Test with various Gemini-generated design types

## Medium Priority
- [ ] Add batch processing optimizations
- [ ] Add processing time logging
- [ ] Add optional debug mode with intermediate outputs

## Low Priority
- [ ] Support for BiRefNet_HR (2K resolution)
