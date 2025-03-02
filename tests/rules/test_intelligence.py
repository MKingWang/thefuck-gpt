import pytest
from unittest.mock import patch, MagicMock
from thefuck.rules.intelligence import match, get_new_command, is_safe_suggestion
from thefuck.types import Command

@pytest.mark.parametrize('command, settings_value, expected', [
    (Command('ls', 'ls: command not found'), 'TRUE', True),
    (Command('fuck', 'not important'), 'TRUE', False),
    (Command('thefuck', 'not important'), 'TRUE', False),
    (Command('ls', 'ls: command not found'), 'FALSE', False),
])
def test_match(command, settings_value, expected):
    with patch('thefuck.rules.intelligence.settings') as settings_mock:
        settings_mock.intelligence = settings_value
        assert match(command) == expected

@patch('thefuck.rules.intelligence.get_fix_suggestion')
def test_get_new_command(get_fix_mock):
    # 测试成功返回安全命令
    get_fix_mock.return_value = "ls -la"
    command = Command('ls', 'ls: command not found')
    assert get_new_command(command) == ["ls -la"]
    
    # 测试返回空命令
    get_fix_mock.return_value = ""
    assert get_new_command(command) == []
    
    # 测试返回不安全的命令
    get_fix_mock.return_value = "rm -rf /"
    assert get_new_command(command) == []

@pytest.mark.parametrize('suggestion, expected', [
    ('ls -la', True),
    ('', False),
    ('rm -rf /tmp/file', False),
    ('sudo rm something', False),
    ('chmod 777 file.txt', False),
    ('echo "Hello" > /dev/null', False),
    ('mv file.txt /', False),
    ('command |', False),
    ('grep "something" file.txt', True),
])
def test_is_safe_suggestion(suggestion, expected):
    assert is_safe_suggestion(suggestion) == expected