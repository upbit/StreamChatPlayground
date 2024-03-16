from typing import Any, Optional
from langchain.schema import LLMResult
from streamlit.external.langchain import StreamlitCallbackHandler, LLMThoughtLabeler
from streamlit.delta_generator import DeltaGenerator


class StreamSpeakCallbackHandler(StreamlitCallbackHandler):
    def __init__(
        self,
        parent_container: DeltaGenerator,
        *,
        max_thought_containers: int = 4,
        expand_new_thoughts: bool = True,
        collapse_completed_thoughts: bool = True,
        thought_labeler: Optional[LLMThoughtLabeler] = None,
    ):
        super().__init__(
            parent_container,
            max_thought_containers=max_thought_containers,
            expand_new_thoughts=expand_new_thoughts,
            collapse_completed_thoughts=collapse_completed_thoughts,
            thought_labeler=thought_labeler,
        )
        self.new_sentence = ""

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        self._require_current_thought().on_llm_end(response, **kwargs)

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self._require_current_thought().on_llm_new_token(token, **kwargs)

        self.new_sentence += token
        # Check if the new token forms a sentence.
        if token in ".:!?。：！？\n" and len(self.new_sentence) > 4:
            try:
                print(">>", self.new_sentence)
            finally:
                self.new_sentence = ""
