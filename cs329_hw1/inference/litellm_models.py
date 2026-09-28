import os
import threading
import time
from typing import List, Dict, Union
import litellm
from litellm import completion, stream_chunk_builder
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Silence LiteLLM's "Provider List: ..." / "Give Feedback" banners on errors.
litellm.suppress_debug_info = True

# Per-model overrides for request parameters.
MODEL_PARAM_OVERRIDES = {
    # gpt-oss reasons before answering and its reasoning counts toward max_tokens,
    # so we use low effort and a larger budget to avoid empty (truncated) answers.
    "azure_ai/gpt-oss-120b": {
        "reasoning_effort": "low",
        "max_tokens": 8192,
        # LiteLLM rejects reasoning_effort for azure_ai/ unless explicitly allowed.
        "allowed_openai_params": ["reasoning_effort"],
    },
    # Non-thinking mode, closest to the original Qwen3-Next Instruct setup.
    # It reasons in the visible answer instead, which often exceeds 4096 tokens.
    "azure_ai/DeepSeek-V4-Flash-0731": {
        "extra_body": {"thinking": {"type": "disabled"}},
        "max_tokens": 8192,
    },
}


class LiteLLMModel:
    """
    A class to send multiple requests to a specified model concurrently using threading.
    """

    def __init__(
        self,
        model: str,
        system_prompt: str = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        max_workers: int = 256,
    ):
        """
        Initializes the LiteLLMModel with the specified model.

        Args:
            model (str): The name of the model to send requests to.
            system_prompt (str): The system prompt to use for the model.
            temperature (float): The temperature to use for the model.
            max_tokens (int): The maximum number of tokens to use for the model.
            max_workers (int): The maximum number of concurrent requests to the model.

        Raises:
            ValueError: If TOGETHER_API_KEY environment variable is not set.
        """
        if not os.getenv("TOGETHER_API_KEY"):
            raise ValueError("TOGETHER_API_KEY environment variable must be set.")

        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.lock = threading.Lock()
        self.max_workers = max_workers

    @retry(wait=wait_random_exponential(min=5, max=10), stop=stop_after_attempt(3))
    def _make_completion_request(self, messages: List[Dict[str, str]]) -> str:
        """
        Makes a completion request with retry logic.

        Args:
            messages (List[Dict[str, str]]): The messages to send to the model.

        Returns:
            str: The response from the model.
        """
        params = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **MODEL_PARAM_OVERRIDES.get(self.model, {}),
        }
        response = completion(
            model=self.model,
            messages=messages,
            # Some providers (e.g. Together's Qwen3.8-Flash) only support streaming.
            stream=True,
            stream_options={"include_usage": True},
            **params,
        )
        # Reassemble the streamed chunks into a single response.
        response = stream_chunk_builder(list(response))
        return response["choices"][0]["message"]["content"]

    def send_request(self, prompt: str) -> str:
        """
        Sends a single request to the model and returns the response.

        Args:
            prompt (str): The prompt to send to the model.

        Returns:
            str: The response from the model or an error message.
        """
        messages = [{"content": prompt, "role": "user"}]
        if self.system_prompt:
            messages.insert(0, {"content": self.system_prompt, "role": "system"})
        try:
            return self._make_completion_request(messages)
        except Exception as e:
            import traceback

            print(f"Error in send_request: {str(e)}")
            print(f"Traceback:\n{traceback.format_exc()}")
            return f"Error: {str(e)}"

    def send_requests(self, prompts: List[str]) -> List[str]:
        """
        Sends multiple requests to the model concurrently and returns the list of responses.
        Uses a thread pool to limit concurrent requests.
        """
        responses = [None] * len(prompts)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for i, prompt in enumerate(prompts):
                # Add small delay between request submissions to avoid rate limits
                if i > 0:
                    time.sleep(0.12)  # ~8.3 requests per second max
                future = executor.submit(self.send_request, prompt)
                futures.append((i, future))

            with tqdm(total=len(prompts), desc="Processing requests") as progress_bar:
                for i, future in futures:
                    responses[i] = future.result()
                    progress_bar.update(1)

        return responses

    def __call__(self, prompts: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Allows the instance to be called as a function to send prompts.

        Args:
            prompts (str or List[str]): A prompt or a list of prompts to send to the model.

        Returns:
            str or List[str]: The response(s) from the model.
        """
        if isinstance(prompts, str):
            return self.send_request(prompts)
        elif isinstance(prompts, list):
            if not all(isinstance(p, str) for p in prompts):
                raise ValueError("All prompts must be strings.")
            return self.send_requests(prompts)
        else:
            raise TypeError("prompts must be a string or a list of strings.")
