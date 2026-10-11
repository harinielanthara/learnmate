from abc import ABC, abstractmethod


class LLMProvider(ABC):
    name: str

    @abstractmethod
    def generate(self, messages: list[dict], temperature: float = 0.1,
                 max_tokens: int = 512, json_mode: bool = False) -> str:
        """messages: [{'role': 'system'|'user'|'assistant', 'content': str}]"""
