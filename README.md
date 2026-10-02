---
title: The Quick Plot — AI Video Translator
emoji: 🎬
colorFrom: yellow
colorTo: blue
sdk: docker
app_port: 7860
---

# The Quick Plot — AI Video Translator

Current build includes:
- Direct video upload
- Public YouTube / Facebook / TikTok / Threads video URL input via yt-dlp
- Gemini automatic transcription and natural conversational Myanmar translation
- Gemini Myanmar TTS voice library and speaking styles
- Myanmar subtitle editor: drag position, X/Y, font size, scale X/Y, alignment, color, outline, shadow, background
- Video colour grading controls
- Built-in original procedural background music: soft piano, acoustic guitar, DJ chill beat
- Background music volume mixing under the generated voice
- Optional automatic 16:9 AI thumbnail generation with Gemini 3.1 Flash Image
- MP4 export and thumbnail download

## Secrets
Set `GEMINI_API_KEY` in Hugging Face Space Settings → Secrets.

Optional model overrides:
- `GEMINI_TEXT_MODEL`
- `GEMINI_TTS_MODEL`
- `GEMINI_TRANSCRIBE_MODEL`
- `GEMINI_IMAGE_MODEL`

The bundled background music files are original procedural audio generated for this app; no external music library is required.
