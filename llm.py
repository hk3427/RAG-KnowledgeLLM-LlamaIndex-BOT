import logging
import os
from pathlib import Path
from llama_index.llms import MockLLM
from llama_index.llms.base import LLM
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import completion_to_prompt,messages_to_prompt

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
logger = logging.getLogger(__name__)

class LLMComponent:
    def __init__(self, llm_mode, llm_model_path=None) -> None:
        self.llm_mode = llm_mode
        self.llm_model_path = llm_model_path
        self.llm = None  # Initialize to None
        logger.info("Initializing the LLM in mode=%s", self.llm_mode)
        self._initialize_llm()

    def _initialize_llm(self):
        if self.llm_mode == "local":
            from llama_index.llms import LlamaCPP

            self.llm = LlamaCPP(
                model_url=None,
                model_path=self.llm_model_path,
                temperature=0.0,
                max_new_tokens=256,
                context_window=3900,
                generate_kwargs={},
                messages_to_prompt=messages_to_prompt,
                completion_to_prompt=completion_to_prompt
            )
        elif self.llm_mode == "openai":
            from llama_index.llms import OpenAI
            self.llm = OpenAI(temperature=0.0, api_key=OPENAI_API_KEY)
        elif self.llm_mode == "mock":
            self.llm = MockLLM()  # MockLLM for testing purposes