"""Tests that enforce module dependency layering.

Loop must not import from syrin.agent except syrin.agent._run_context
(AgentRunContext). Bottom/mid must not import top (agent, cli).
"""

from __future__ import annotations

import ast
from pathlib import Path


def _collect_imports_from_file(path: Path) -> list[tuple[str, str | None]]:
    """Parse a Python file and return (module, name) for each from X import Y / import X."""
    out: list[tuple[str, str | None]] = []
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return out
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.append((alias.name, None))
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.append((node.module, None))
    return out


def test_loop_imports_only_agent_run_context() -> None:
    """Loop must depend only on AgentRunContext from agent, not on Agent or agent/__init__."""
    # Repo root: tests/unit/architecture/test_layering.py -> parent x4
    root = Path(__file__).resolve().parent.parent.parent.parent
    loop_py = root / "src" / "syrin" / "loop.py"
    assert loop_py.exists(), "loop.py not found"
    imports = _collect_imports_from_file(loop_py)
    agent_imports = [m for m, _ in imports if "syrin.agent" in m]
    assert set(agent_imports) <= {"syrin.agent._run_context"}, (
        "loop.py may only import from syrin.agent._run_context (AgentRunContext), "
        f"got: {agent_imports}"
    )


def test_no_agent_import_in_bottom_mid_modules() -> None:
    """Bottom and mid layers must not import top (agent, cli)."""
    root = Path(__file__).resolve().parent.parent.parent.parent
    src = root / "src" / "syrin"
    # Bottom: enums, exceptions, types, domain_events, config
    # Mid: cost, budget, budget_store, threshold, events, response, prompt, tool, task, output, validation, pipe
    # Upper (may import mid/bottom but not agent): model, providers, loop, memory, context, guardrails, checkpoint, ratelimit, observability
    # Top: agent
    bottom_mid = {
        "enums",
        "exceptions",
        "types",
        "domain_events",
        "config",
        "cost",
        "budget",
        "budget_store",
        "threshold",
        "events",
        "response",
        "prompt",
        "tool",
        "task",
        "output",
        "validation",
        "pipe",
    }
    top_modules = {"syrin.agent"}
    # Check main __init__.py of each package (not submodules) for simplicity
    for name in bottom_mid:
        pkg = src / name
        if not pkg.is_dir():
            pkg = src / f"{name}.py"
        if not pkg.exists():
            continue
        if not pkg.is_dir():
            path = pkg
        elif not (pkg / "__init__.py").exists():
            continue
        else:
            path = pkg / "__init__.py"
        imports = _collect_imports_from_file(path)
        for mod, _ in imports:
            for top in top_modules:
                if mod == top or mod.startswith(top + "."):
                    # Exclude agent._run_context and agent._context_builder (internal, not "top" surface)
                    if "agent._run_context" in mod or "agent._context_builder" in mod:
                        continue
                    raise AssertionError(
                        f"{path.relative_to(root)} must not import top layer {mod} "
                        "(dependency direction: bottom/mid do not import agent, cli)"
                    )
