import os
import sqlite3
import json
import requests
from datetime import datetime, timedelta
from typing import Optional, Any
import sys
import traceback
from thefuck.conf import settings

class LLMCache:
    """Local SQLite caching implementation."""
    def __init__(self, ttl_hours: int = 24):
        self.conn = sqlite3.connect(os.path.expanduser('~/.thefuck_llm_cache.db'))
        self._init_db()
        self.ttl = timedelta(hours=ttl_hours)

    def _init_db(self):
        with self.conn:
            self.conn.execute('''CREATE TABLE IF NOT EXISTS cache
                                (hash TEXT PRIMARY KEY,
                                 suggestion TEXT,
                                 created_at TIMESTAMP)''')

    def get(self, prompt_hash: str) -> Optional[str]:
        cursor = self.conn.execute('''SELECT suggestion, created_at FROM cache
                                    WHERE hash=?''', (prompt_hash,))
        row = cursor.fetchone()
        if row:
            suggestion, created_at = row
            if datetime.now() - datetime.fromisoformat(created_at) < self.ttl:
                return suggestion
        return None

    def set(self, prompt_hash: str, suggestion: str):
        with self.conn:
            self.conn.execute('''INSERT OR REPLACE INTO cache
                               VALUES (?, ?, ?)''',
                            (prompt_hash, suggestion, datetime.now().isoformat()))

class LLMFactory:
    _supported_providers = {
        'openai': {
            'model_configs': {
                'model': lambda: os.environ.get('OPENAI_MODEL', settings.openai_model),
                'api_key': lambda: os.environ.get('OPENAI_API_KEY', settings.openai_api_key),
                'base_url': lambda: os.environ.get('OPENAI_BASE_URL', settings.openai_base_url),
                'temperature': lambda: float(os.environ.get('THEFUCK_TEMPERATURE', settings.temperature)),
            },
            'api_endpoint': lambda base_url: f"{base_url.rstrip('/')}/v1/chat/completions"
        },
        'siliconflow': {
            'model_configs': {
                'model': lambda: os.environ.get('SILICONFLOW_MODEL', settings.siliconflow_model),
                'api_key': lambda: os.environ.get('SILICONFLOW_API_KEY', settings.siliconflow_api_key),
                'temperature': lambda: float(os.environ.get('THEFUCK_TEMPERATURE', settings.temperature)),
                'base_url': lambda: "https://api.siliconflow.cn/"
            },
            'api_endpoint': lambda base_url: f"{base_url.rstrip('/')}/v1/chat/completions",
            'openai_compatible': True
        },
        'deepseek': {
            'model_configs': {
                'model': lambda: os.environ.get('DEEPSEEK_MODEL', settings.deepseek_model),
                'api_key': lambda: os.environ.get('DEEPSEEK_API_KEY', settings.deepseek_api_key),
                'temperature': lambda: float(os.environ.get('THEFUCK_TEMPERATURE', settings.temperature)),
                'base_url': lambda: "https://api.deepseek.com"
            },
            'api_endpoint': lambda base_url: f"{base_url.rstrip('/')}/v1/chat/completions"
        },
        'ollama': {
            'model_configs': {
                'model': lambda: os.environ.get('OLLAMA_MODEL', settings.ollama_model),
                'temperature': lambda: float(os.environ.get('THEFUCK_TEMPERATURE', settings.temperature)),
                'base_url': lambda: os.environ.get('OLLAMA_BASE_URL', settings.ollama_base_url),
            },
            'api_endpoint': lambda base_url: f"{base_url.rstrip('/')}/api/chat"
        }
    }

    @classmethod
    def create_payload_and_headers(cls, provider, config, messages):
        """创建请求负载和头部"""
        if provider == "ollama":
            return {
                "payload": {
                    "model": config.get("model"),
                    "messages": messages,
                    "temperature": config.get("temperature", 0.7)
                },
                "headers": {
                    "Content-Type": "application/json"
                }
            }
        else:  # OpenAI 兼容的API
            return {
                "payload": {
                    "model": config.get("model"),
                    "messages": messages,
                    "temperature": config.get("temperature", 0.7)
                },
                "headers": {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {config.get('api_key')}"
                }
            }

    @classmethod
    def get_provider_info(cls, provider_name):
        provider_name = provider_name.lower()
        if provider_name not in cls._supported_providers:
            raise ValueError(f"Unsupported LLM provider: {provider_name}")

        provider_config = cls._supported_providers[provider_name]
        config = {
            key: getter() 
            for key, getter in provider_config['model_configs'].items()
        }
        
        # 处理OpenAI兼容API
        api_provider = provider_name
        if provider_config.get('openai_compatible', False):
            api_provider = "openai"
            
        api_endpoint = provider_config['api_endpoint'](config.get('base_url', ''))
        
        return api_provider, config, api_endpoint

def call_llm_api(provider, endpoint, payload, headers):
    """调用LLM API并解析结果"""
    try:
        response = requests.post(endpoint, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        result = response.json()
        
        # 解析不同供应商的响应
        if provider == "ollama":
            return result.get("message", {}).get("content", "")
        else:  # OpenAI 兼容的API
            return result.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception as e:
        print(f"API调用失败: {str(e)}", file=sys.stderr)
        raise

def generate_llm_response(context: Any, user_prompt: str, system_prompts: list, use_cache: bool = True, is_fix_mode: bool = False) -> str:
    """
    Generic LLM response generation function, supporting custom prompts and caching
    
    Args:
        context: Context content for response generation
        system_prompts: List of system prompts
        user_prompt: User prompt template
        use_cache: Whether to use cache, default is True
        is_fix_mode: Whether it's in fix mode
        
    Returns:
        str: Response generated by LLM
    """
    # 格式化提示
    formatted_user_prompt = user_prompt.format(context=context)
    
    # Cache handling
    if use_cache:
        cache = LLMCache(24)
        # Generate unique cache key, including prompts and context
        cache_data = {
            "context": context,
            "system_prompts": system_prompts,
            "user_prompt": user_prompt
        }
        context_str = json.dumps(cache_data, sort_keys=True)
        prompt_hash = str(hash(context_str))
        
        # Check cache
        if cached := cache.get(prompt_hash):
            return cached
    
    # 构建消息数组
    messages = []
    for prompt in system_prompts:
        messages.append({"role": "system", "content": prompt})
    messages.append({"role": "user", "content": formatted_user_prompt})
    
    try:
        # 获取LLM提供商信息
        provider = os.environ.get('THEFUCK_MODEL_PROVIDER', settings.model_provider)
        provider, config, endpoint = LLMFactory.get_provider_info(provider)
        
        # 构建请求数据
        request_data = LLMFactory.create_payload_and_headers(provider, config, messages)
        
        print("Generating...", flush=True, file=sys.stderr)
        response = call_llm_api(provider, endpoint, request_data["payload"], request_data["headers"])
        
        # Cache results
        if use_cache:
            cache.set(prompt_hash, response)
            
        return response
        
    except Exception as e:
        traceback.print_exc()
        print(f"LLM call failed: {str(e)}", flush=True, file=sys.stderr)
        if is_fix_mode:
            print("AI-based suggestion is not available. Falling back to non-AI mode.", flush=True, file=sys.stderr)
        return ""
    
def get_fix_suggestion(context: Any) -> str:
    """
    Generate command correction suggestions
    """
    system_prompts = [
        "Please only return the corrected command without any explanation. Ensure the command is safe and compliant with Unix standards.",
        "Only a single command is needed, not multiple commands.",
        "If the error message contains 'command not found', generate an approximate common command.",
        "If the command itself is not an installation command, such as apt, apt-get, or yum, do not generate an installation command.",
        "Ensure that the command output is clean, without any additional information such as markdown formatting, prompts, or comments."
    ]
    user_prompt = "Please help me fix the following command: {context}"

    return generate_llm_response(context, user_prompt, system_prompts)

def generate_command(context: str) -> str:
    """
    Generate command from natural language description
    """
    system_prompts = [
        "You are a Linux command line expert.",
        "Generate the most suitable shell command based on the user's natural language request.",
        "Return only the command itself, without any explanations or other text.",
        "Ensure that the command output is clean, without any additional information such as markdown formatting, prompts, or comments.",
        "Ensure the command is valid and follows best practices."
    ]
    user_prompt = "Generate a command to: {context}"
    
    return generate_llm_response(context, user_prompt, system_prompts)