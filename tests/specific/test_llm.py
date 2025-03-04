import os
import json
import pytest
from thefuck.specific.llm import (
    generate_llm_response,
    LLMFactory,
    LLMCache,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class DummyChain:
    def __or__(self, other):
        return self
    def invoke(self, input_dict):
        return "dummy response"

@pytest.fixture(autouse=True)
def patch_llm_components(monkeypatch):
    monkeypatch.setattr(
        ChatPromptTemplate, "from_messages", lambda messages: DummyChain()
    )
    monkeypatch.setattr(
        StrOutputParser, "__new__", lambda cls, *args, **kwargs: DummyChain()
    )
    monkeypatch.setattr(LLMFactory, "create", classmethod(lambda cls: DummyChain()))

@pytest.mark.parametrize("provider,expected_model_provider", [
    ("siliconflow", "openai"), 
    ("ollama", "ollama"), 
])
def test_generate_llm_response_providers(monkeypatch, provider, expected_model_provider, tmp_path):
    monkeypatch.setenv("THEFUCK_MODEL_PROVIDER", provider)
    
    monkeypatch.setenv("THEFUCK_TEMPERATURE", "0.7")
    
    monkeypatch.setenv("SILICONFLOW_MODEL", "deepseek-ai/DeepSeek-V3")
    monkeypatch.setenv("SILICONFLOW_API_KEY", "sk-ouxkxd")
    
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:32b")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://192.168.10.14:11434")
    
    context = "List all files in the current directory." 
    system_prompts = [
        "You are a Linux command line expert.",
        "Generate the most suitable shell command based on the user's natural language request.",
        "Return only the command itself, without any explanations or other text.",
        "Ensure that the command output is clean, without any additional information such as markdown formatting, prompts, or comments.",
        "Ensure the command is valid and follows best practices."
    ]
    user_prompt = "Generate a command to: {context}"
    
    monkeypatch.setenv("HOME", str(tmp_path))
    
    response = generate_llm_response(context, user_prompt, system_prompts, use_cache=False)
    
    print("Generated Response (no cache):", response)
    
    assert response == "dummy response"
    
    response_cache = generate_llm_response(context, user_prompt, system_prompts, use_cache=True)
    print("Generated Response (with cache):", response_cache)
    assert response_cache == "dummy response"
    
    cache_data = {
        "context": context,
        "system_prompts": system_prompts,
        "user_prompt": user_prompt
    }
    context_str = json.dumps(cache_data, sort_keys=True)
    prompt_hash = str(hash(context_str))
    
    cache_instance = LLMCache(24)
    cached = cache_instance.get(prompt_hash)
    print("Cached Response:", cached)
    assert cached == "dummy response"