# note 模型工厂，根据模型名返回模型实例
from functools import cache

from langchain_anthropic import ChatAnthropic
from langchain_aws import ChatBedrock
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from core.settings import settings
from schema.models import (
    AllModelEnum,
    AnthropicModelName,
    AWSModelName,
    AzureOpenAIModelName,
    DeepseekModelName,
    MiMoModelName,
    FakeModelName,
    GoogleModelName,
    GroqModelName,
    OllamaModelName,
    OpenAICompatibleName,
    OpenAIModelName,
    OpenRouterModelName,
    VertexAIModelName,
)

_MODEL_TABLE = (
    {m: m.value for m in OpenAIModelName}
    | {m: m.value for m in OpenAICompatibleName}
    | {m: m.value for m in AzureOpenAIModelName}
    | {m: m.value for m in DeepseekModelName}
    | {m: m.value for m in MiMoModelName}
    | {m: m.value for m in AnthropicModelName}
    | {m: m.value for m in GoogleModelName}
    | {m: m.value for m in VertexAIModelName}
    | {m: m.value for m in GroqModelName}
    | {m: m.value for m in AWSModelName}
    | {m: m.value for m in OllamaModelName}
    | {m: m.value for m in OpenRouterModelName}
    | {m: m.value for m in FakeModelName}
)


class FakeToolModel(FakeListChatModel):
    def __init__(self, responses: list[str]):
        super().__init__(responses=responses)

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


type ModelT = (
    AzureChatOpenAI
    | ChatOpenAI
    | ChatAnthropic
    | ChatGoogleGenerativeAI
    | ChatGroq
    | ChatBedrock
    | ChatOllama
    | FakeToolModel
)

# tip 这里有些模型不需要设置base_url 和 api_key，因为langchain对他们的支持更好，函数底层会自动读取.env中的配置
@cache
def get_model(model_name: AllModelEnum, /) -> ModelT:
    # 设置了 streaming=True 的模型会逐 token 实时发送生成内容
    # 前提是调用 /stream 接口时 stream_tokens=True（默认值）
    api_model_name = _MODEL_TABLE.get(model_name)
    if not api_model_name:
        raise ValueError(f"Unsupported model: {model_name}")

    if model_name in OpenAIModelName:
        return ChatOpenAI(model=api_model_name, streaming=True)
    if model_name in OpenAICompatibleName:
        if not settings.COMPATIBLE_BASE_URL or not settings.COMPATIBLE_MODEL:
            raise ValueError("OpenAICompatible base url and endpoint must be configured")

        return ChatOpenAI(
            model=settings.COMPATIBLE_MODEL,
            temperature=0.5,
            streaming=True,
            openai_api_base=settings.COMPATIBLE_BASE_URL,
            openai_api_key=settings.COMPATIBLE_API_KEY,
        )
    if model_name in AzureOpenAIModelName:
        if not settings.AZURE_OPENAI_API_KEY or not settings.AZURE_OPENAI_ENDPOINT:
            raise ValueError("Azure OpenAI API key and endpoint must be configured")

        return AzureChatOpenAI(
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            deployment_name=api_model_name,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            streaming=True,
            timeout=60,
            max_retries=3,
        )
    # tip 这里的两个国内模型也是用ChatOpenAI()，但是不显示设置base和key的话，就会去访问openAI了
    if model_name in DeepseekModelName:
        return ChatOpenAI(
            model=api_model_name,
            temperature=0.5,
            streaming=True,
            openai_api_base="https://api.deepseek.com",
            openai_api_key=settings.DEEPSEEK_API_KEY,
        )
    if model_name in MiMoModelName:
        return ChatOpenAI(
            model=api_model_name,
            streaming=True,
            base_url=settings.MIMO_BASE_URL,
            api_key=settings.MIMO_API_KEY,
            extra_body={
                "thinking": {
                    "type": "disabled",
                }
            },
        )
    if model_name in AnthropicModelName:
        if model_name == AnthropicModelName.SONNET_5:
            return ChatAnthropic(model_name=api_model_name, streaming=True)
        return ChatAnthropic(model_name=api_model_name, temperature=0.5, streaming=True)
    if model_name in GoogleModelName:
        return ChatGoogleGenerativeAI(model=api_model_name, temperature=0.5, streaming=True)
    if model_name in GroqModelName:
        if model_name == GroqModelName.GPT_OSS_SAFEGUARD_20B:
            return ChatGroq(model=api_model_name, temperature=0.0)  # type: ignore[call-arg]
        return ChatGroq(model=api_model_name, temperature=0.5)  # type: ignore[call-arg]
    if model_name in AWSModelName:
        if model_name == AWSModelName.BEDROCK_SONNET:
            return ChatBedrock(model=api_model_name)
        return ChatBedrock(model=api_model_name, temperature=0.5)
    if model_name in OllamaModelName:
        if not settings.OLLAMA_MODEL:
            raise ValueError("Ollama model must be configured")
        if settings.OLLAMA_BASE_URL:
            chat_ollama = ChatOllama(
                model=settings.OLLAMA_MODEL, temperature=0.5, base_url=settings.OLLAMA_BASE_URL
            )
        else:
            chat_ollama = ChatOllama(model=settings.OLLAMA_MODEL, temperature=0.5)
        return chat_ollama
    if model_name in OpenRouterModelName:
        if not settings.OPENROUTER_API_KEY:
            raise ValueError("OpenRouter API key must be configured")
        return ChatOpenAI(
            model=api_model_name,
            temperature=0.5,
            streaming=True,
            base_url="https://openrouter.ai/api/v1/",
            api_key=settings.OPENROUTER_API_KEY,
        )
    if model_name in FakeModelName:
        return FakeToolModel(responses=["This is a test response from the fake model."])

    raise ValueError(f"Unsupported model: {model_name}")
