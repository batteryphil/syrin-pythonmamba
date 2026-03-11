"""Cartesia Sonic TTS provider. Requires cartesia package.

Ultra-low latency (~90ms TTFB). Best for real-time voice agents.
"""

from __future__ import annotations

from typing import Any

from syrin.generation._base_voice import BaseVoiceProvider


class CartesiaVoiceProvider(BaseVoiceProvider):
    """Cartesia Sonic TTS provider. Implements VoiceGenerationProvider.

    Requires: pip install syrin[voice] or pip install cartesia.
    Set CARTESIA_API_KEY or pass api_key=.
    """

    def __init__(
        self,
        api_key: str | None = None,
        voice_id: str = "default",
        model: str = "sonic-3",
        **kwargs: Any,
    ) -> None:
        super().__init__(api_key=api_key, model=model, **kwargs)
        self.voice_id = voice_id

    def _get_api_key_env(self) -> str:
        return "CARTESIA_API_KEY"

    def _get_package_name(self) -> str:
        return "syrin[voice]"

    def _synthesize(
        self,
        text: str,
        *,
        api_key: str,
        voice_id: str,
        speed: float,
        language: str,
        output_format: str,
        model_id: str,
        **kwargs: Any,
    ) -> tuple[bytes, str, str]:
        from cartesia import Cartesia

        vid = voice_id if voice_id != "default" else self.voice_id
        client = Cartesia(api_key=api_key)
        response = client.tts.generate(
            model_id=model_id,
            transcript=text,
            voice_id=vid,
            output_format=output_format,
            **{**self._kwargs, **kwargs},
        )
        content_bytes = response if isinstance(response, bytes) else bytes(response)
        mime = "audio/mpeg" if output_format == "mp3" else f"audio/{output_format}"
        return content_bytes, "sonic-3", mime
