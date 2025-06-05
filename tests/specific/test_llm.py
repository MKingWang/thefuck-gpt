import os
import json
import pytest
from unittest.mock import patch, Mock
from thefuck.specific.llm import (
    generate_llm_response,
    LLMFactory,
    LLMCache,
    call_llm_api
)

@pytest.fixture(autouse=True)
def patch_llm_components(monkeypatch):
    # 模拟API调用返回固定结果
    def mock_call_llm_api(*args, **kwargs):
        return "dummy response"
    
    monkeypatch.setattr('thefuck.specific.llm.call_llm_api', mock_call_llm_api)

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
    
    # 测试不使用缓存的情况
    with patch('thefuck.specific.llm.LLMFactory.get_provider_info') as mock_provider_info:
        mock_provider_info.return_value = (expected_model_provider, {}, "https://api.example.com/v1/completions")
        with patch('thefuck.specific.llm.LLMFactory.create_payload_and_headers') as mock_create_payload:
            mock_create_payload.return_value = {
                "payload": {}, 
                "headers": {}
            }
            
            response = generate_llm_response(context, user_prompt, system_prompts, use_cache=False)
            
            print("Generated Response (no cache):", response)
            assert response == "dummy response"
            
            # 验证调用参数
            if provider == "siliconflow":
                mock_provider_info.assert_called_with("siliconflow")
            else:
                mock_provider_info.assert_called_with("ollama")
    
    # 测试使用缓存的情况
    with patch('thefuck.specific.llm.LLMFactory.get_provider_info') as mock_provider_info:
        mock_provider_info.return_value = (expected_model_provider, {}, "https://api.example.com/v1/completions")
        with patch('thefuck.specific.llm.LLMFactory.create_payload_and_headers') as mock_create_payload:
            mock_create_payload.return_value = {
                "payload": {}, 
                "headers": {}
            }
            
            response_cache = generate_llm_response(context, user_prompt, system_prompts, use_cache=True)
            print("Generated Response (with cache):", response_cache)
            assert response_cache == "dummy response"
    
    # 验证缓存
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

def test_call_llm_api():
    # 测试 OpenAI 兼容的 API
    with patch('requests.post') as mock_post:
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "ls -la"}}]
        }
        mock_post.return_value = mock_response
        
        result = call_llm_api(
            "openai", 
            "https://api.example.com/v1/chat/completions",
            {"model": "gpt-3.5-turbo", "messages": []},
            {"Authorization": "Bearer test"}
        )
        
        assert result == "ls -la"
        mock_post.assert_called_once()
    
    # 测试 Ollama API
    with patch('requests.post') as mock_post:
        mock_response = Mock()
        mock_response.json.return_value = {
            "message": {"content": "ls -la"}
        }
        mock_post.return_value = mock_response
        
        result = call_llm_api(
            "ollama", 
            "http://localhost:11434/api/chat",
            {"model": "llama2", "messages": []},
            {"Content-Type": "application/json"}
        )
        
        assert result == "ls -la"
        mock_post.assert_called_once()