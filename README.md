---
title: CutoutAI Background Remover
emoji: 🖼️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# CutoutAI Background Removal Service

This space hosts a custom API for removing backgrounds from images using BiRefNet.

## API Usage

**Endpoint:** `/remove-background`

**Method:** `POST`

**Body:** `multipart/form-data` with an `image` file.

**Response:** Returns the image PNG with transparent background.
