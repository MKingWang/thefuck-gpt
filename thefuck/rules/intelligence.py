import os
import re
from datetime import datetime
from thefuck.specific.llm import get_fix_suggestion
from thefuck.types import Command
from thefuck.conf import settings

# Dangerous command pattern filtering
DANGEROUS_PATTERNS = [
    r'rm\s+-\w*f',       # Force deletion
    r'sudo\s+rm',        # sudo deletion
    r'chmod\s+[0-7]{3,4}\s+',  # Permission modification
    r'>\s*/dev/null',    # Output redirection
    r'mv\s+.*\s+/',      # Moving files to the root directory
    r'(\||&)\s*$'        # Command injection symbols
]

def match(command: Command) -> bool:
    """Match all commands not handled by other rules."""
    if settings.intelligence.upper() != "ENABLED":
        return False
    return command.script_parts[0] not in ['fuck', 'thefuck']

def get_new_command(command: Command) -> list[str]:
    """Generate a new command using an enhanced prompt context."""
    context = {
        "system_info": {
            "os": os.name,
            "shell": os.getenv('SHELL'),
            "python_version": os.getenv('PYTHON_VERSION')
        },
        "current_command": command.script,
        "error_output": command.output,
        "timestamp": datetime.now().isoformat()
    }

    suggestion = get_fix_suggestion(context)
    
    if suggestion and is_safe_suggestion(suggestion):
        return [suggestion] 
    else:
        return []

    
def is_safe_suggestion(suggestion: str) -> bool:
    """Check if the suggested command is safe."""
    suggestion = suggestion.strip()

    # Check for empty command
    if not suggestion:
        return False
        
    # Dangerous pattern matching
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, suggestion, re.IGNORECASE):
            return False
        
    return True

priority = 10