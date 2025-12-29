# Background Removal Tool - Project Assets

> **Purpose**: Self-hosted background removal API for Etsy t-shirt workflow

---

## Core Files

| File | Description |
|------|-------------|
| `cutoutai.py` | Core BiRefNet processing (365 lines) |
| `api.py` | FastAPI server with webhooks (351 lines) |
| `Dockerfile` | Production container |
| `requirements.txt` | Python dependencies |
| `test_cutout.py` | Automated test script |

---

## Configuration

| File | Description |
|------|-------------|
| `PROMPT.md` | Ralph development instructions |
| `@fix_plan.md` | Task priority tracking |
| `specs/requirements.md` | Technical specifications |

---

## Test Outputs

| File | Description |
|------|-------------|
| `test_output.png` | Synthetic test result |
| `real_test_output.png` | cosmic_bloom.png result |
| `hard_test_output.png` | ChatGPT image result (3.6MB input) |

---

## Key Features

- **Models**: matting, general, portrait, lite, hr, dynamic
- **API**: REST + Webhook (n8n compatible)
- **Output**: PNG, base64
- **Thresholding**: 0.2 (capture_all) / 0.4 (standard)

---

## Deployment Status

| Target | Status |
|--------|--------|
| Local | ✅ Ready |
| Railway | ⬜ Not deployed |
| HuggingFace | ⬜ Not deployed |

---

## Related Projects

| Project | Relationship |
|---------|--------------|
| `etsy tshirt project` | Primary consumer of this API |
| `system-instructions` | CCR/Ralph configuration |

---

*Last Updated: Dec 28, 2025*
