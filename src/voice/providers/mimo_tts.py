from __future__ import annotations

import base64
import binascii
import os
from typing import Literal

import httpx

AudioFormat = Literal["wav", "mp3"]


class MiMoTTSError(RuntimeError):
    """MiMo TTS 调用失败。"""


class MiMoTTS:
    MIME_TYPES = {
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
    }

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        voice: str | None = None,
        audio_format: AudioFormat | None = None,
        timeout: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ):
        self.api_key = api_key or os.getenv("MIMO_API_KEY", "")
        self.base_url = (
            base_url
            or os.getenv("MIMO_BASE_URL")
            or "https://api.xiaomimimo.com/v1"
        ).rstrip("/")
        self.model = (
            model
            or os.getenv("VOICE_TTS_MODEL")
            or "mimo-v2.5-tts"
        )
        self.voice = (
            voice
            or os.getenv("VOICE_TTS_VOICE")
            or "mimo_default"
        )
        self.audio_format = (
            audio_format
            or os.getenv("VOICE_TTS_FORMAT")
            or "wav"
        )
        self.timeout = timeout
        self.client = client

        if not self.api_key:
            raise ValueError("未配置 MIMO_API_KEY")

        if self.audio_format not in self.MIME_TYPES:
            raise ValueError(
                "VOICE_TTS_FORMAT 只能是 wav 或 mp3"
            )

    async def synthesize(
        self,
        text: str,
        voice: str | None = None,
        audio_format: AudioFormat | None = None,
        style: str | None = None,
    ) -> bytes:
        prepared_text = text.strip()

        if not prepared_text:
            raise ValueError("待合成文本不能为空")

        selected_voice = (voice or self.voice).strip()
        selected_format = audio_format or self.audio_format

        if not selected_voice:
            raise ValueError("TTS 音色不能为空")

        if selected_format not in self.MIME_TYPES:
            raise ValueError("MiMo TTS 仅支持 wav 和 mp3 格式")

        messages: list[dict[str, str]] = []

        if style and style.strip():
            messages.append(
                {
                    "role": "user",
                    "content": style.strip(),
                }
            )

        messages.append(
            {
                "role": "assistant",
                "content": prepared_text,
            }
        )

        payload = {
            "model": self.model,
            "messages": messages,
            "audio": {
                "format": selected_format,
                "voice": selected_voice,
            },
            "stream": False,
        }

        response = await self._post(payload)

        try:
            response_data = response.json()
            audio_base64 = (
                response_data["choices"][0]["message"]["audio"]["data"]
            )
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise MiMoTTSError(
                "MiMo TTS 返回的数据格式不正确"
            ) from exc

        if not isinstance(audio_base64, str) or not audio_base64:
            raise MiMoTTSError("MiMo TTS 未返回音频数据")

        # 同时兼容纯 Base64 和 Data URL。
        if audio_base64.startswith("data:"):
            _, separator, audio_base64 = audio_base64.partition(",")

            if not separator:
                raise MiMoTTSError("MiMo TTS 返回的 Data URL 不正确")

        audio_base64 = "".join(audio_base64.split())

        try:
            audio_data = base64.b64decode(
                audio_base64,
                validate=True,
            )
        except (ValueError, binascii.Error) as exc:
            raise MiMoTTSError(
                "无法解码 MiMo TTS 返回的音频"
            ) from exc

        if not audio_data:
            raise MiMoTTSError("MiMo TTS 返回了空音频")

        return audio_data

    def get_mime_type(
        self,
        audio_format: AudioFormat | None = None,
    ) -> str:
        selected_format = audio_format or self.audio_format

        if selected_format not in self.MIME_TYPES:
            raise ValueError("音频格式只能是 wav 或 mp3")

        return self.MIME_TYPES[selected_format]

    async def _post(self, payload: dict) -> httpx.Response:
        url = f"{self.base_url}/chat/completions"

        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json",
        }

        try:
            if self.client is not None:
                response = await self.client.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
            else:
                async with httpx.AsyncClient(
                    timeout=self.timeout
                ) as client:
                    response = await client.post(
                        url,
                        headers=headers,
                        json=payload,
                    )
        except httpx.TimeoutException as exc:
            raise MiMoTTSError("MiMo TTS 请求超时") from exc
        except httpx.RequestError as exc:
            raise MiMoTTSError(
                f"无法连接 MiMo TTS 服务：{exc}"
            ) from exc

        if response.is_error:
            detail = self._extract_error(response)
            raise MiMoTTSError(
                f"MiMo TTS 请求失败，HTTP {response.status_code}：{detail}"
            )

        return response

    @staticmethod
    def _extract_error(response: httpx.Response) -> str:
        try:
            data = response.json()

            if isinstance(data, dict):
                error = data.get("error")

                if isinstance(error, dict):
                    detail = (
                        error.get("message")
                        or error.get("detail")
                        or error
                    )
                else:
                    detail = (
                        data.get("message")
                        or data.get("detail")
                        or data
                    )
            else:
                detail = data

            return str(detail)[:500]
        except ValueError:
            return response.text[:500] or "未知错误"