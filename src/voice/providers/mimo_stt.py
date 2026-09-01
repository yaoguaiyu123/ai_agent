from __future__ import annotations

import base64
import os
from typing import Literal

import httpx

AudioFormat = Literal["wav", "mp3"]
Language = Literal["auto", "zh", "en"]


class MiMoSTTError(RuntimeError):
    """MiMo ASR 调用失败。"""


class MiMoSTT:
    MIME_TYPES = {
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
    }

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        language: Language | None = None,
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
            or os.getenv("VOICE_STT_MODEL")
            or "mimo-v2.5-asr"
        )
        self.language = (
            language
            or os.getenv("VOICE_STT_LANGUAGE")
            or "auto"
        )
        self.timeout = timeout
        self.client = client

        if not self.api_key:
            raise ValueError("未配置 MIMO_API_KEY")

        if self.language not in {"auto", "zh", "en"}:
            raise ValueError(
                "VOICE_STT_LANGUAGE 只能是 auto、zh 或 en"
            )

    async def transcribe(
        self,
        audio_data: bytes,
        audio_format: AudioFormat = "wav",
        language: Language | None = None,
    ) -> str:
        if not audio_data:
            raise ValueError("音频数据不能为空")

        if audio_format not in self.MIME_TYPES:
            raise ValueError("MiMo ASR 仅支持 wav 和 mp3 格式")

        selected_language = language or self.language

        if selected_language not in {"auto", "zh", "en"}:
            raise ValueError("language 只能是 auto、zh 或 en")

        mime_type = self.MIME_TYPES[audio_format]
        encoded_audio = base64.b64encode(audio_data).decode("ascii")
        data_url = f"data:{mime_type};base64,{encoded_audio}"

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": data_url,
                                "format": audio_format,
                            },
                        }
                    ],
                }
            ],
            "asr_options": {
                "language": selected_language,
            },
            "stream": False,
        }

        response = await self._post(payload)

        try:
            response_data = response.json()
            text = response_data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise MiMoSTTError(
                "MiMo ASR 返回的数据格式不正确"
            ) from exc

        if not isinstance(text, str):
            raise MiMoSTTError("MiMo ASR 返回的识别结果不是字符串")

        text = text.strip()

        if not text:
            raise MiMoSTTError("MiMo ASR 未返回识别文本")

        return text

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
            raise MiMoSTTError("MiMo ASR 请求超时") from exc
        except httpx.RequestError as exc:
            raise MiMoSTTError(
                f"无法连接 MiMo ASR 服务：{exc}"
            ) from exc

        if response.is_error:
            detail = self._extract_error(response)
            raise MiMoSTTError(
                f"MiMo ASR 请求失败，HTTP {response.status_code}：{detail}"
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