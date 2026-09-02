from enum import StrEnum, auto
# note 枚举类，定义了各家大语言模型的版本

class Provider(StrEnum):
    OPENAI = auto()
    OPENAI_COMPATIBLE = auto()
    AZURE_OPENAI = auto()
    DEEPSEEK = auto()
    ANTHROPIC = auto()
    GOOGLE = auto()
    VERTEXAI = auto()
    GROQ = auto()
    AWS = auto()
    OLLAMA = auto()
    OPENROUTER = auto()
    FAKE = auto()
    MIMO = auto()


class OpenAIModelName(StrEnum):
    GPT_5_NANO = "gpt-5-nano"
    GPT_5_MINI = "gpt-5-mini"
    GPT_5_1 = "gpt-5.1"
    GPT_56_LUNA = "gpt-5.6-luna"
    GPT_56_TERRA = "gpt-5.6-terra"
    GPT_56_SOL = "gpt-5.6-sol"


class AzureOpenAIModelName(StrEnum):
    AZURE_GPT_5 = "azure-gpt-5"
    AZURE_GPT_5_MINI = "azure-gpt-5-mini"


class DeepseekModelName(StrEnum):
    DEEPSEEK_V4_FLASH = "deepseek-v4-flash"
    DEEPSEEK_V4_PRO = "deepseek-v4-pro"

class MiMoModelName(StrEnum):
    """Xiaomi MiMo文字模型"""

    MIMO_V25 = "mimo-v2.5"
    MIMO_V25_PRO = "mimo-v2.5-pro"

class AnthropicModelName(StrEnum):

    HAIKU_45 = "claude-haiku-4-5"
    SONNET_45 = "claude-sonnet-4-5"
    SONNET_5 = "claude-sonnet-5"


class GoogleModelName(StrEnum):

    GEMINI_25_PRO = "gemini-2.5-pro"
    GEMINI_31_FLASH_LITE = "gemini-3.1-flash-lite"
    GEMINI_35_FLASH = "gemini-3.5-flash"
    GEMINI_35_FLASH_LITE = "gemini-3.5-flash-lite"
    GEMINI_36_FLASH = "gemini-3.6-flash"
    GEMINI_31_PRO_PREVIEW = "gemini-3.1-pro-preview"


class VertexAIModelName(StrEnum):
    GEMINI_25_PRO = "models/gemini-2.5-pro"
    GEMINI_31_FLASH_LITE = "models/gemini-3.1-flash-lite"
    GEMINI_35_FLASH = "models/gemini-3.5-flash"
    GEMINI_35_FLASH_LITE = "models/gemini-3.5-flash-lite"
    GEMINI_36_FLASH = "models/gemini-3.6-flash"
    GEMINI_31_PRO_PREVIEW = "models/gemini-3.1-pro-preview"


class GroqModelName(StrEnum):

    GPT_OSS_20B = "openai/gpt-oss-20b"
    GPT_OSS_120B = "openai/gpt-oss-120b"
    GPT_OSS_SAFEGUARD_20B = "openai/gpt-oss-safeguard-20b"


class AWSModelName(StrEnum):

    BEDROCK_HAIKU = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
    BEDROCK_SONNET = "global.anthropic.claude-sonnet-5"


class OllamaModelName(StrEnum):

    OLLAMA_GENERIC = "ollama"


class OpenRouterModelName(StrEnum):
    GEMINI_35_FLASH = "google/gemini-3.5-flash"
    GEMINI_36_FLASH = "google/gemini-3.6-flash"


class OpenAICompatibleName(StrEnum):
    OPENAI_COMPATIBLE = "openai-compatible"


class FakeModelName(StrEnum):
    FAKE = "fake"


type AllModelEnum = (
    OpenAIModelName
    | OpenAICompatibleName
    | AzureOpenAIModelName
    | DeepseekModelName
    | AnthropicModelName
    | GoogleModelName
    | VertexAIModelName
    | GroqModelName
    | AWSModelName
    | OllamaModelName
    | OpenRouterModelName
    | FakeModelName
    | MiMoModelName
)
