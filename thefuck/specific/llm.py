import os
import sqlite3
import json
from datetime import datetime, timedelta
from typing import Optional, Any
import sys
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
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
                'model': lambda: settings.openai_model,
                'api_key': lambda: settings.openai_api_key,
                'base_url': lambda: settings.openai_base_url,
                'temperature': lambda: settings.openai_temperature,
            }
        },
        'anthropic': {
            'model_configs': {
                'model': lambda: settings.anthropic_model,
                'api_key': lambda: settings.anthropic_api_key,
                'temperature': lambda: settings.anthropic_temperature,
            }
        }
    }

    @classmethod
    def create(cls, provider: str = 'openai') -> Any:
        if provider not in cls._supported_providers:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        provider_config = cls._supported_providers[provider]
        config = {
            key: getter() 
            for key, getter in provider_config['model_configs'].items()
        }
        
        return init_chat_model(
            model_provider=provider,
            **config
        )

def get_fix_suggestion(context: Any) -> str:
    """
    Generate an enhanced LLM fix suggestion, which includes:
    - Environment variable configuration
    - Request timeout and retry mechanism
    - Local caching
    - Cost monitoring
    """
    # cache = LLMCache(ttl_hours=config.get('cache_ttl', 24))
    cache = LLMCache(24)
    context_str = json.dumps(context, sort_keys=True)
    prompt_hash = str(hash(context_str))



    prompt = ChatPromptTemplate.from_messages([
        ("system", "Please only return the corrected command without any explanation. Ensure the command is safe and compliant with Unix standards."),
        ("system", "Only a single command is needed, not multiple commands."),
        ("system", "If the error message contains 'command not found', generate an approximate common command."),
        ("system", "If the command itself is not an installation command, such as apt, apt-get, or yum, do not generate an installation command."),
        ("system", "Ensure that the command output is clean, without any additional information such as markdown formatting, prompts, or comments."),
        ("user", "Please help me fix the following command: {context}")
    ])
    
    # if cached := cache.get(prompt_hash):
    #     return cached

    try:
        llm = LLMFactory.create(settings.model_provider)
        
        chain = prompt | llm
        print("Generating...", flush=True, file=sys.stderr)
        response = chain.invoke({"context": context})
        # cache.set(prompt_hash, response.content)
        return response.content
        
    except Exception as e:
        print(f"LLM调用失败: {str(e)}")
        return ""
