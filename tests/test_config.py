"""Tests for config module"""
import os
import pytest
from pathlib import Path
from open_agent.config import get_base_url, get_model, PROVIDER_DEFAULTS, load_config_file


def test_get_base_url_explicit():
    """Test explicit base_url has highest priority"""
    url = get_base_url(base_url="http://custom:8080/v1")
    assert url == "http://custom:8080/v1"


def test_get_base_url_env_var(monkeypatch):
    """Test environment variable is used when no explicit URL"""
    monkeypatch.setenv("OPEN_AGENT_BASE_URL", "http://env-server:1234/v1")
    url = get_base_url()
    assert url == "http://env-server:1234/v1"


def test_get_base_url_explicit_overrides_env(monkeypatch):
    """Test explicit URL overrides environment variable"""
    monkeypatch.setenv("OPEN_AGENT_BASE_URL", "http://env-server:1234/v1")
    url = get_base_url(base_url="http://explicit:8080/v1")
    assert url == "http://explicit:8080/v1"


def test_get_base_url_provider_lmstudio():
    """Test provider default for LM Studio"""
    url = get_base_url(provider="lmstudio")
    assert url == "http://localhost:1234/v1"


def test_get_base_url_provider_ollama():
    """Test provider default for Ollama"""
    url = get_base_url(provider="ollama")
    assert url == "http://localhost:11434/v1"


def test_get_base_url_provider_llamacpp():
    """Test provider default for llama.cpp"""
    url = get_base_url(provider="llamacpp")
    assert url == "http://localhost:8080/v1"


def test_get_base_url_provider_vllm():
    """Test provider default for vLLM"""
    url = get_base_url(provider="vllm")
    assert url == "http://localhost:8000/v1"


def test_get_base_url_provider_case_insensitive():
    """Test provider name is case-insensitive"""
    url = get_base_url(provider="OLLAMA")
    assert url == "http://localhost:11434/v1"


def test_get_base_url_provider_unknown():
    """Test unknown provider falls back to default"""
    url = get_base_url(provider="unknown-provider")
    assert url == PROVIDER_DEFAULTS["lmstudio"]


def test_get_base_url_default():
    """Test default is LM Studio when nothing specified"""
    url = get_base_url()
    assert url == "http://localhost:1234/v1"


def test_get_base_url_explicit_overrides_provider():
    """Test explicit URL overrides provider default"""
    url = get_base_url(
        base_url="http://custom:8080/v1",
        provider="ollama"
    )
    assert url == "http://custom:8080/v1"


def test_get_base_url_env_overrides_provider(monkeypatch):
    """Test environment variable overrides provider default"""
    monkeypatch.setenv("OPEN_AGENT_BASE_URL", "http://env-server:1234/v1")
    url = get_base_url(provider="ollama")
    assert url == "http://env-server:1234/v1"


def test_load_config_file_no_yaml():
    """Test load_config_file returns empty dict when YAML not installed"""
    # This will work even without YAML installed since it catches ImportError
    config = load_config_file(Path("/nonexistent/path/config.yaml"))
    assert isinstance(config, dict)


def test_load_config_file_nonexistent():
    """Test load_config_file returns empty dict for nonexistent file"""
    config = load_config_file(Path("/nonexistent/path/config.yaml"))
    assert config == {}


def test_load_config_file_valid_yaml(tmp_path):
    """Test load_config_file reads and parses valid YAML"""
    pytest.importorskip("yaml")  # Skip if PyYAML not installed

    config_file = tmp_path / "test-config.yaml"
    config_file.write_text("""
base_url: http://localhost:1234/v1
model: qwen2.5-32b-instruct
temperature: 0.7
max_tokens: 4096
""")

    config = load_config_file(config_file)

    assert config["base_url"] == "http://localhost:1234/v1"
    assert config["model"] == "qwen2.5-32b-instruct"
    assert config["temperature"] == 0.7
    assert config["max_tokens"] == 4096


def test_load_config_file_empty_yaml(tmp_path):
    """Test load_config_file returns empty dict for empty YAML file"""
    pytest.importorskip("yaml")

    config_file = tmp_path / "empty.yaml"
    config_file.write_text("")

    config = load_config_file(config_file)
    assert config == {}


def test_load_config_file_search_path_priority(tmp_path, monkeypatch):
    """Test search paths are checked in priority order"""
    pytest.importorskip("yaml")

    # Change to tmp directory
    monkeypatch.chdir(tmp_path)

    # Create config in position 2 (should be found but not used)
    config_dir = tmp_path / ".config" / "open-agent"
    config_dir.mkdir(parents=True)
    (config_dir / "config.yaml").write_text("model: from-xdg\n")

    # Create config in position 1 (should win)
    (tmp_path / "open-agent.yaml").write_text("model: from-project\n")

    config = load_config_file()

    # Project config (priority 1) should win
    assert config["model"] == "from-project"


def test_load_config_file_fallback_to_second_path(tmp_path, monkeypatch):
    """Test fallback to second search path when first doesn't exist"""
    pytest.importorskip("yaml")

    monkeypatch.chdir(tmp_path)
    # Mock home directory to be tmp_path so XDG path resolves correctly
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    # Only create config in position 2 (XDG)
    config_dir = tmp_path / ".config" / "open-agent"
    config_dir.mkdir(parents=True)
    (config_dir / "config.yaml").write_text("model: from-xdg\n")

    config = load_config_file()

    # Should fall back to XDG config
    assert config["model"] == "from-xdg"


def test_config_file_with_env_override_pattern(tmp_path, monkeypatch):
    """Test documented pattern: config file + env var override"""
    pytest.importorskip("yaml")

    # Create config file
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
base_url: http://config-server:1234/v1
model: config-model
""")

    # Set environment variable (should override model)
    monkeypatch.setenv("OPEN_AGENT_MODEL", "env-model")

    # Follow documented pattern
    config = load_config_file(config_file)
    model = get_model(config.get("model"))  # Env > config
    base_url = get_base_url(config.get("base_url"))  # No env, use config

    # Env var should override config for model
    assert model == "env-model"
    # Config value should be used for base_url (no env var)
    assert base_url == "http://config-server:1234/v1"


def test_load_config_file_invalid_yaml_raises(tmp_path):
    """Test load_config_file raises yaml.YAMLError on invalid YAML"""
    yaml = pytest.importorskip("yaml")

    config_file = tmp_path / "bad.yaml"
    config_file.write_text("""
base_url: http://localhost:1234/v1
model: [unclosed bracket
temperature: 0.7
""")

    # Invalid YAML should raise, not return {}
    # This is consistent with documented behavior: errors are not caught
    with pytest.raises(yaml.YAMLError):
        load_config_file(config_file)


def test_load_config_file_fallback_to_home_dotfile(tmp_path, monkeypatch):
    """Test fallback to ~/.open-agent.yaml (third search path)"""
    pytest.importorskip("yaml")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    # Only create ~/.open-agent.yaml (third priority)
    (tmp_path / ".open-agent.yaml").write_text("model: from-home-dotfile\n")

    config = load_config_file()

    # Should fall back to home dotfile
    assert config["model"] == "from-home-dotfile"


def test_load_config_file_null_yaml_returns_empty(tmp_path):
    """Test load_config_file handles YAML null/None properly"""
    pytest.importorskip("yaml")

    config_file = tmp_path / "null.yaml"
    config_file.write_text("null\n")  # Valid YAML that returns None

    config = load_config_file(config_file)

    # yaml.safe_load(null) returns None, should convert to {}
    assert config == {}


def test_load_config_file_explicit_path_only(tmp_path, monkeypatch):
    """Test explicit config_path ignores default search paths"""
    pytest.importorskip("yaml")

    monkeypatch.chdir(tmp_path)

    # Create default search path with one config
    (tmp_path / "open-agent.yaml").write_text("model: from-search-path\n")

    # Create explicit path with different config
    explicit = tmp_path / "custom.yaml"
    explicit.write_text("model: from-explicit\n")

    config = load_config_file(explicit)

    # Should use explicit path, not search path
    assert config["model"] == "from-explicit"


def test_load_config_file_unreadable_raises(tmp_path):
    """Test load_config_file raises on permission errors (documented behavior)"""
    pytest.importorskip("yaml")

    config_file = tmp_path / "unreadable.yaml"
    config_file.write_text("model: test\n")

    # Remove read permissions
    import os
    os.chmod(config_file, 0o000)

    try:
        # Should raise PermissionError as documented
        # "File I/O errors are NOT caught - will raise if file exists but unreadable"
        with pytest.raises(PermissionError):
            load_config_file(config_file)
    finally:
        # Cleanup: restore permissions so file can be deleted
        os.chmod(config_file, 0o644)


def test_provider_defaults_exist():
    """Test that all expected providers have defaults"""
    assert "lmstudio" in PROVIDER_DEFAULTS
    assert "ollama" in PROVIDER_DEFAULTS
    assert "llamacpp" in PROVIDER_DEFAULTS
    assert "vllm" in PROVIDER_DEFAULTS


def test_provider_defaults_format():
    """Test that provider defaults are valid URLs"""
    for provider, url in PROVIDER_DEFAULTS.items():
        assert url.startswith("http://") or url.startswith("https://")
        assert "/v1" in url


# Model configuration tests


def test_get_model_returns_fallback_when_env_missing(monkeypatch):
    """Fallback parameter should be used when env var is absent."""
    monkeypatch.delenv("OPEN_AGENT_MODEL", raising=False)
    model = get_model(model="qwen2.5-32b-instruct")
    assert model == "qwen2.5-32b-instruct"


def test_get_model_env_var_overrides_fallback(monkeypatch):
    """Environment variable should override provided fallback by default."""
    monkeypatch.setenv("OPEN_AGENT_MODEL", "llama3.1:70b")
    model = get_model(model="qwen2.5-32b-instruct")
    assert model == "llama3.1:70b"


def test_get_model_can_ignore_env(monkeypatch):
    """prefer_env=False should force the fallback model."""
    monkeypatch.setenv("OPEN_AGENT_MODEL", "llama3.1:70b")
    model = get_model(model="qwen2.5-32b-instruct", prefer_env=False)
    assert model == "qwen2.5-32b-instruct"


def test_get_model_none_when_not_set(monkeypatch):
    """Test returns None when nothing specified."""
    monkeypatch.delenv("OPEN_AGENT_MODEL", raising=False)
    model = get_model()
    assert model is None
