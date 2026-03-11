# Multimodal Examples

Examples covering every aspect of Syrin’s multimodal features: input (images, files, PDF), capabilities (`input_media` / `output_media` / `input_file_rules`), generation (standalone and agent tools), and vision routing.

## Run

From the project root:

```bash
uv run python -m examples.18_multimodal.<script_name>
```

Or with `python` if the package is installed:

```bash
python -m examples.18_multimodal.<script_name>
```

## Examples

| Script | What it covers |
|--------|----------------|
| **multimodal_image_text.py** | Content parts (text + image_url), `file_to_message`, vision-capable agent |
| **file_to_message** | Built into multimodal_image_text: building data URLs from file bytes for content parts |
| **standalone_generate_image.py** | `generate_image()` without an agent; `GenerationResult`; error when key missing; optional save |
| **standalone_generate_video.py** | `generate_video()` without an agent; polling; `GenerationResult`; error when key missing |
| **generation_with_options.py** | `AspectRatio` and `OutputMimeType` StrEnums with standalone `generate_image` |
| **agent_with_generation_tools.py** | `output_media={Media.IMAGE, Media.VIDEO}`; agent gets generate_image/generate_video tools when `GOOGLE_API_KEY` set |
| **agent_explicit_generators.py** | `ImageGenerator.Gemini()` / `.DALLE()`, `VideoGenerator.Veo()`; explicit generators override defaults |
| **agent_input_media_output_media.py** | Declaring `input_media` and `output_media`; capability discovery (`_input_media`, `_output_media`) |
| **agent_file_input_rules.py** | `Media.FILE` in `input_media` + `InputFileRules` (allowed_mime_types, max_size_mb); validation when FILE without rules |
| **vision_routing_multimodal.py** | Router + profiles with `input_media`; ModalityDetector; routing to vision profile when message contains image |
| **pdf_extract_example.py** | `pdf_extract_text()` (optional `syrin[pdf]`); sending extracted text to agent; `file_to_message` for PDF data URL |

## Requirements

- **Image + text input, content parts, routing**: vision-capable model (e.g. OpenAI `gpt-4o`) or Almock for structure. Set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` for real models.
- **Standalone image/video generation**: `pip install syrin[generation]`, `GOOGLE_API_KEY` (or `GEMINI_API_KEY`).
- **PDF text extraction**: `pip install syrin[pdf]`

## Feature map

| Feature | Example(s) |
|---------|------------|
| Content parts (text + image_url) | multimodal_image_text, vision_routing_multimodal |
| `file_to_message` | multimodal_image_text, pdf_extract_example, vision_routing_multimodal |
| `pdf_extract_text` | pdf_extract_example |
| `Media` enum | agent_input_media_output_media, agent_file_input_rules, agent_with_generation_tools, vision_routing_multimodal |
| `input_media` / `output_media` | agent_input_media_output_media, agent_with_generation_tools, vision_routing_multimodal |
| `InputFileRules` | agent_file_input_rules |
| Standalone `generate_image` / `generate_video` | standalone_generate_image, standalone_generate_video, generation_with_options |
| Agent generation tools (output_media) | agent_with_generation_tools |
| Explicit image_generation / video_generation (Gemini, DALLE) | agent_explicit_generators |
| Vision routing (ModalityDetector, input_media) | vision_routing_multimodal |
| AspectRatio / OutputMimeType | generation_with_options |
