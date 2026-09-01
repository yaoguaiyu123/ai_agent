from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field

from voice.providers.mimo_stt import (
    AudioFormat,
    Language,
    MiMoSTT,
    MiMoSTTError,
)
from voice.providers.mimo_tts import (
    MiMoTTS,
    MiMoTTSError,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/voice",
    tags=["voice"],
)

MIME_TO_AUDIO_FORMAT: dict[str, AudioFormat] = {
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
}

DEFAULT_MAX_AUDIO_BYTES = 25 * 1024 * 1024


class TranscriptionResponse(BaseModel):
    text: str
    model: str
    language: Language


class SpeechSynthesisInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    text: str = Field(
        min_length=1,
        description="需要转换为语音的文本",
    )
    voice: str | None = Field(
        default=None,
        description="MiMo预置音色",
        examples=["mimo_default", "冰糖", "Mia"],
    )
    audio_format: Literal["wav", "mp3"] | None = Field(
        default=None,
        alias="format",
        description="输出音频格式",
    )
    style: str | None = Field(
        default=None,
        description="语气、情绪和语速描述",
        examples=["自然、亲切，语速适中"],
    )


@lru_cache(maxsize=1)
def get_stt_provider() -> MiMoSTT:
    return MiMoSTT()


@lru_cache(maxsize=1)
def get_tts_provider() -> MiMoTTS:
    return MiMoTTS()


def get_max_audio_bytes() -> int:
    value = os.getenv(
        "VOICE_MAX_AUDIO_BYTES",
        str(DEFAULT_MAX_AUDIO_BYTES),
    )

    try:
        maximum = int(value)
    except ValueError:
        logger.warning(
            "VOICE_MAX_AUDIO_BYTES配置错误，使用默认值：%s",
            DEFAULT_MAX_AUDIO_BYTES,
        )
        return DEFAULT_MAX_AUDIO_BYTES

    if maximum <= 0:
        return DEFAULT_MAX_AUDIO_BYTES

    return maximum


def check_content_length(
    request: Request,
    maximum: int,
) -> None:
    content_length = request.headers.get("content-length")

    if not content_length:
        return

    try:
        size = int(content_length)
    except ValueError:
        return

    if size > maximum:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"音频文件过大，最大允许{maximum}字节",
        )


@router.post(
    "/transcribe",
    response_model=TranscriptionResponse,
    summary="将语音转换为文本",
)
async def transcribe(
    request: Request,
    language: Language = Query(
        default="auto",
        description="识别语言：auto、zh或en",
    ),
    audio_format: AudioFormat | None = Query(
        default=None,
        description="音频格式，可通过查询参数指定wav或mp3",
    ),
) -> TranscriptionResponse:
    maximum = get_max_audio_bytes()
    check_content_length(request, maximum)

    content_type = (
        request.headers
        .get("content-type", "")
        .split(";", maxsplit=1)[0]
        .strip()
        .lower()
    )

    detected_format = MIME_TO_AUDIO_FORMAT.get(content_type)
    selected_format = audio_format or detected_format

    if selected_format is None:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                "无法确定音频格式，请使用audio/wav、audio/mpeg，"
                "或者通过audio_format查询参数指定wav或mp3"
            ),
        )

    audio_data = await request.body()

    if not audio_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="音频数据不能为空",
        )

    if len(audio_data) > maximum:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"音频文件过大，最大允许{maximum}字节",
        )

    try:
        stt = get_stt_provider()

        text = await stt.transcribe(
            audio_data=audio_data,
            audio_format=selected_format,
            language=language,
        )
    except ValueError as exc:
        logger.warning("MiMo STT配置或输入错误：%s", exc)

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except MiMoSTTError as exc:
        logger.error("MiMo STT调用失败：%s", exc)

        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    return TranscriptionResponse(
        text=text,
        model=stt.model,
        language=language,
    )


@router.post(
    "/synthesize",
    summary="将文本转换为语音",
    responses={
        200: {
            "content": {
                "audio/wav": {},
                "audio/mpeg": {},
            },
            "description": "生成的音频二进制数据",
        }
    },
)
async def synthesize(
    synthesis_input: SpeechSynthesisInput,
) -> Response:
    try:
        tts = get_tts_provider()

        selected_format = (
            synthesis_input.audio_format
            or tts.audio_format
        )

        audio_data = await tts.synthesize(
            text=synthesis_input.text,
            voice=synthesis_input.voice,
            audio_format=selected_format,
            style=synthesis_input.style,
        )

        mime_type = tts.get_mime_type(selected_format)
    except ValueError as exc:
        logger.warning("MiMo TTS配置或输入错误：%s", exc)

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except MiMoTTSError as exc:
        logger.error("MiMo TTS调用失败：%s", exc)

        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    filename = f"speech.{selected_format}"

    return Response(
        content=audio_data,
        media_type=mime_type,
        headers={
            "Content-Disposition": (
                f'inline; filename="{filename}"'
            ),
            "X-Audio-Format": selected_format,
            "Cache-Control": "no-store",
        },
    )