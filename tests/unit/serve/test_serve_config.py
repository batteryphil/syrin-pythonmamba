"""Tests for ServeConfig dataclass."""

from __future__ import annotations

from syrin.enums import ServeProtocol
from syrin.serve.config import ServeConfig


def test_serve_config_defaults() -> None:
    """ServeConfig has correct defaults."""
    config = ServeConfig()
    assert config.protocol == ServeProtocol.HTTP
    assert config.host == "0.0.0.0"
    assert config.port == 8000
    assert config.route_prefix == ""
    assert config.stream is True
    assert config.include_metadata is True
    assert config.debug is False
    assert config.enable_playground is False
    assert config.enable_discovery is None


def test_serve_config_explicit_values() -> None:
    """ServeConfig accepts explicit values."""
    config = ServeConfig(
        protocol=ServeProtocol.CLI,
        host="127.0.0.1",
        port=9000,
        route_prefix="/api/v1",
        stream=False,
        include_metadata=False,
        debug=True,
        enable_playground=True,
        enable_discovery=False,
    )
    assert config.protocol == ServeProtocol.CLI
    assert config.host == "127.0.0.1"
    assert config.port == 9000
    assert config.route_prefix == "/api/v1"
    assert config.stream is False
    assert config.include_metadata is False
    assert config.debug is True
    assert config.enable_playground is True
    assert config.enable_discovery is False


def test_serve_config_stdio_protocol() -> None:
    """ServeConfig with STDIO protocol."""
    config = ServeConfig(protocol=ServeProtocol.STDIO)
    assert config.protocol == ServeProtocol.STDIO


def test_serve_config_enable_discovery_none_auto() -> None:
    """enable_discovery=None means auto-detect."""
    config = ServeConfig(enable_discovery=None)
    assert config.enable_discovery is None


def test_serve_config_enable_discovery_false() -> None:
    """enable_discovery=False forces discovery off."""
    config = ServeConfig(enable_discovery=False)
    assert config.enable_discovery is False


def test_serve_config_enable_discovery_true() -> None:
    """enable_discovery=True forces discovery on."""
    config = ServeConfig(enable_discovery=True)
    assert config.enable_discovery is True


def test_serve_config_route_prefix() -> None:
    """route_prefix is preserved."""
    config = ServeConfig(route_prefix="/agent/v1")
    assert config.route_prefix == "/agent/v1"


def test_serve_config_port_validation() -> None:
    """ServeConfig accepts valid port (no validation for now)."""
    config = ServeConfig(port=3000)
    assert config.port == 3000
