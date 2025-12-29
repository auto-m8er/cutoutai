# CutoutAI - Background Remover

An enhanced, flawless background removal tool built on BiRefNet for perfect t-shirt mockup preparation.

## Features

- **Flawless Removal**: No patchy faces, artifacts, or edge issues
- **Multi-Element Capture**: Captures bubbles, decorations, and all design elements
- **API Ready**: Webhook, HTTP API, terminal commands
- **Cloud Hosted**: Designed for n8n, Make, and cloud automation
- **Mockup Quality**: Optimized for Printify t-shirt mockups

## Quick Start

```python
from cutoutai import remove_background

# Basic usage
result = remove_background("design.png")
result.save("design_cutout.png")

# With enhanced settings for complex designs
result = remove_background(
    "design.png",
    capture_all_elements=True,  # Get bubbles, small elements
    edge_refinement=True,       # Smooth edges
    matting_mode="general"      # or "portrait" for faces
)
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/remove` | POST | Remove background from image |
| `/api/v1/batch` | POST | Process multiple images |
| `/api/v1/health` | GET | Health check |
| `/webhook` | POST | n8n/Make webhook endpoint |

## Workflow Integration

### n8n Webhook
```
POST https://your-host/webhook
Content-Type: multipart/form-data

image: <file>
options: {"capture_all_elements": true}
```

### CLI
```bash
cutoutai process design.png --output cutout.png
cutoutai batch ./designs/ --output ./cutouts/
```

## Quality Settings

| Setting | Description | Use Case |
|---------|-------------|----------|
| `capture_all_elements` | Detect and preserve small elements (bubbles, decorations) | Complex designs |
| `edge_refinement` | Smooth and feather edges | All mockups |
| `matting_mode` | `general`, `portrait`, or `heavy` | Match content type |
| `output_resolution` | Preserve or scale output | Printify requirements |

## License

MIT License - Built on BiRefNet
