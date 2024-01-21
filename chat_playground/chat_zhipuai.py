"""ZHIPU AI chat models wrapper."""
from __future__ import annotations

import logging
import os
import sys
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    agenerate_from_stream,
    generate_from_stream,
)
from langchain_core.language_models.llms import create_base_retry_decorator
from langchain_core.messages import (
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessageChunk,
    FunctionMessageChunk,
    HumanMessageChunk,
    SystemMessageChunk,
    ToolMessageChunk,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from langchain_core.runnables import Runnable
from langchain_core.utils import (
    get_from_dict_or_env,
    get_pydantic_field_names,
)
from langchain_community.adapters.openai import (
    convert_dict_to_message,
    convert_message_to_dict,
)

logger = logging.getLogger(__name__)


class ChatZhipuAI(BaseChatModel):
    """`ZhipuAI` Chat large language models API.

    To use, you should have the ``zhipuai`` python package installed, and the
    environment variable ``ZHIPU_API_KEY`` set with your API key.

    Any parameters that are valid to be passed to the openai.create call can be passed
    in, even if not explicitly saved on this class.

    Example:
        .. code-block:: python

            from langchain_community.chat_models import ChatZhipuAI
            llm = ChatZhipuAI(model_name="glm-4")
    """

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model_name": self.model_name}

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "zhipuai-chat"

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"zhipuai_api_key": "ZHIPUAI_API_KEY"}

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "chat_models", "zhipuai"]

    @property
    def lc_attributes(self) -> Dict[str, Any]:
        attributes: Dict[str, Any] = {}

        if self.model_name:
            attributes["model"] = self.model_name

        if self.streaming:
            attributes["streaming"] = self.streaming

        return attributes

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by Langchain."""
        return True

    client: Any = Field(default=None, exclude=True)  #: :meta private:
    model_name: str = Field(default="glm-4", alias="model")
    """Model name to use."""

    temperature: float = Field(0.9)
    """What sampling temperature to use."""
    top_p: float = Field(0.7)
    """Another method of sampling temperature is called nuclear sampling"""
    streaming: bool = False
    """Whether to stream the results or not."""

    # When updating this to use a SecretStr
    # Check for classes that derive from this class (as some of them
    # may assume zhipuai_api_key is a str)
    zhipuai_api_key: Optional[str] = Field(default=None, alias="api_key")
    """Automatically inferred from env var `ZHIPUAI_API_KEY` if not provided."""

    max_tokens: Optional[int] = None
    """Maximum number of tokens to generate."""

    class Config:
        """Configuration for this pydantic object."""

        allow_population_by_field_name = True

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling OpenAI API."""
        params = {
            "model": self.model_name,
            "stream": self.streaming,
            "temperature": self.temperature,
            # **self.model_kwargs,
        }
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        return params

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.zhipuai_api_key = get_from_dict_or_env(kwargs, "api_key", "ZHIPUAI_API_KEY")
        try:
            import zhipuai

        except ImportError:
            raise ImportError(
                "Could not import zhipuai python package. " "Please install it with `pip install zhipuai`."
            )

        if not self.client:
            self.client = zhipuai.ZhipuAI(api_key=self.zhipuai_api_key)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        should_stream = stream if stream is not None else self.streaming
        if stop and len(stop) > 1:
            # ZhipuAI only support one stop word, cutoff
            # eg: ['\nObservation:', '\n\tObservation:'] -> ['\nObservation:']
            stop = [stop[0]]
        if should_stream:
            stream_iter = self._stream(messages, stop=stop, run_manager=run_manager, **kwargs)
            result = generate_from_stream(stream_iter)

            # Fix ChatResult.llm_output empty, fill token_usage in last message
            finish_message = result.generations[-1]
            if finish_message.generation_info:
                info = finish_message.generation_info
                result.llm_output = {"token_usage": info.get("token_usage"), "model_name": self.model_name}
            return result

        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {
            **params,
            **({"stream": stream} if stream is not None else {}),
            **kwargs,
        }
        response = self.completion(messages=message_dicts, run_manager=run_manager, **params)
        return self._create_chat_result(response)

    def _create_message_dicts(
        self, messages: List[BaseMessage], stop: Optional[List[str]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        params = self._default_params
        if stop is not None:
            if "stop" in params:
                raise ValueError("`stop` found in both the input and default params.")
            params["stop"] = stop
        message_dicts = [convert_message_to_dict(m) for m in messages]
        return message_dicts, params

    def _create_chat_result(self, response: Union[dict, BaseModel]) -> ChatResult:
        generations = []
        if not isinstance(response, dict):
            response = response.dict()
        for res in response["choices"]:
            message = convert_dict_to_message(res["message"])
            generation_info = dict(finish_reason=res.get("finish_reason"))
            # if "logprobs" in res:
            #     generation_info["logprobs"] = res["logprobs"]
            gen = ChatGeneration(
                message=message,
                generation_info=generation_info,
            )
            generations.append(gen)
        token_usage = response.get("usage")
        llm_output = {
            "token_usage": token_usage,
            "model_name": self.model_name,
        }
        return ChatResult(generations=generations, llm_output=llm_output)

    def _combine_llm_outputs(self, llm_outputs: List[Optional[dict]]) -> dict:
        overall_token_usage: dict = {}
        for output in llm_outputs:
            if output is None:
                # Happens in streaming
                continue
            token_usage = output["token_usage"]
            if token_usage is not None:
                for k, v in token_usage.items():
                    if k in overall_token_usage:
                        overall_token_usage[k] += v
                    else:
                        overall_token_usage[k] = v
        combined = {"token_usage": overall_token_usage, "model_name": self.model_name}
        return combined

    def completion(self, run_manager: Optional[AsyncCallbackManagerForLLMRun] = None, **kwargs: Any) -> Any:
        # call zhipuai
        return self.client.chat.completions.create(**kwargs)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, "stream": True}

        default_chunk_class = AIMessageChunk
        for raw_chunk in self.completion(messages=message_dicts, **params):
            if not isinstance(raw_chunk, dict):
                raw_chunk = raw_chunk.dict()
            if len(raw_chunk["choices"]) == 0:
                continue
            choice = raw_chunk["choices"][0]
            chunk = _convert_delta_to_message_chunk(choice["delta"], default_chunk_class)
            finish_reason = choice.get("finish_reason")
            if finish_reason is not None:
                token_usage = raw_chunk.get("usage")  # Get token_usage in finish message
                generation_info = dict(finish_reason=finish_reason, token_usage=token_usage)
            else:
                generation_info = None
            default_chunk_class = chunk.__class__
            chunk = ChatGenerationChunk(message=chunk, generation_info=generation_info)
            yield chunk
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)


def _convert_delta_to_message_chunk(
    _dict: Mapping[str, Any], default_class: Type[BaseMessageChunk]
) -> BaseMessageChunk:
    role = _dict.get("role")
    content = _dict.get("content") or ""
    additional_kwargs: Dict = {}
    if _dict.get("function_call"):
        function_call = dict(_dict["function_call"])
        if "name" in function_call and function_call["name"] is None:
            function_call["name"] = ""
        additional_kwargs["function_call"] = function_call
    if _dict.get("tool_calls"):
        additional_kwargs["tool_calls"] = _dict["tool_calls"]

    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content)
    elif role == "assistant" or default_class == AIMessageChunk:
        return AIMessageChunk(content=content, additional_kwargs=additional_kwargs)
    elif role == "system" or default_class == SystemMessageChunk:
        return SystemMessageChunk(content=content)
    elif role == "function" or default_class == FunctionMessageChunk:
        return FunctionMessageChunk(content=content, name=_dict["name"])
    elif role == "tool" or default_class == ToolMessageChunk:
        return ToolMessageChunk(content=content, tool_call_id=_dict["tool_call_id"])
    elif role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role)
    else:
        return default_class(content=content)
