# -*- coding: utf-8 -*-
"""
@Time    : 2024/4/8 15:15
@Author  : deryzhou
@File    : provider/hunyuan_api.py
"""
import uuid
import json

from metagpt.configs.llm_config import LLMConfig, LLMType
from metagpt.logs import log_llm_stream
from metagpt.provider.base_llm import BaseLLM

# from metagpt.provider.llm_provider_registry import register_provider
from metagpt.utils.cost_manager import TokenCostManager

from metagpt.provider.general_api_requestor import GeneralAPIRequestor
from metagpt.provider.zhipuai.async_sse_client import AsyncSSEClient

HUNYUAN_DEFAULT_TIMEOUT = 3000


# @register_provider([LLMType.HUNYUAN])
class HunyuanAPI(BaseLLM):

    def __init__(self, config: LLMConfig, **kwargs):
        self.__init_hunyuan(config)
        self.client = GeneralAPIRequestor(base_url=config.base_url)
        self.config = config
        self.suffix_url = "/openapi/chat/completions"
        self.http_method = "post"
        self.use_system_prompt = True
        self.cost_manager = TokenCostManager()
        self.wsid = "10103"
        self.kwargs = kwargs

    def __init_hunyuan(self, config: LLMConfig):
        assert config.api_key, "api_key is required!"
        assert config.base_url, "base_url is required!"
        self.model = config.model

    def _const_kwargs(self, messages: list[dict], stream: bool = False) -> dict:
        kwargs = {
            "model": self.model,
            "query_id": "test_query_id_" + str(uuid.uuid4()),
            "messages": messages,
            "repetition_penalty": 1,
            "output_seq_len": 1024,
            "max_input_seq_len": 2048,
            "stream": stream,
        }
        for k, v in self.kwargs.items():
            kwargs[k] = v
        return kwargs

    def _default_headers(self) -> dict:
        return {
            "Connection": "keep-alive",
            "Authorization": "Bearer " + self.config.api_key,
            "Wsid": self.wsid,
        }

    def get_choice_text(self, resp: dict) -> str:
        """get the resp content from llm response"""
        # 默认 'delta': {'role': 'assistant'}
        assist_msg = resp.get("delta", {})
        return assist_msg.get("content")

    def get_usage(self, resp: dict) -> dict:
        return resp.get("usage", {})

    def _decode_and_load(self, chunk: bytes, encoding: str = "utf-8") -> dict:
        chunk = chunk.decode(encoding)
        return json.loads(chunk)

    async def _achat_completion(
        self, messages: list[dict], timeout: int = HUNYUAN_DEFAULT_TIMEOUT
    ) -> dict:
        resp, _, _ = await self.client.arequest(
            method=self.http_method,
            url=self.suffix_url,
            headers=self._default_headers(),
            params=self._const_kwargs(messages),
            request_timeout=self.get_timeout(timeout),
        )
        resp = self._decode_and_load(resp)
        usage = self.get_usage(resp)
        self._update_costs(usage)
        return resp

    async def acompletion(
        self, messages: list[dict], timeout=HUNYUAN_DEFAULT_TIMEOUT
    ) -> dict:
        return await self._achat_completion(messages, timeout=self.get_timeout(timeout))

    async def _achat_completion_stream(
        self, messages: list[dict], timeout: int = HUNYUAN_DEFAULT_TIMEOUT
    ) -> AsyncSSEClient:
        stream_resp, _, _ = await self.client.arequest(
            method=self.http_method,
            url=self.suffix_url,
            headers=self._default_headers(),
            params=self._const_kwargs(messages, stream=True),
            request_timeout=self.get_timeout(timeout),
            stream=True,
        )

        collected_content = []
        usage = {}
        async for chunk in AsyncSSEClient(stream_resp).stream():
            if chunk.get("error", None):
                raise Exception(json.dumps(chunk))

            content = chunk["choices"][0]
            if content.get("finish_reason", None) == "stop":
                usage = self.get_usage(chunk)
                break
            data = self.get_choice_text(content)
            if not data:  # 可能是role类的无content返回，跳过
                continue

            collected_content.append(data)
            log_llm_stream(data)

        log_llm_stream("\n")

        self._update_costs(usage)
        full_content = "".join(collected_content)
        return full_content
