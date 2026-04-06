"""Pry — Rich live dashboard for agent debugging."""

from __future__ import annotations

import contextlib
import io
import json
import queue
import re
import sys
import textwrap
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, ParamSpec, TypedDict, TypeVar, Unpack

FilterMode = Literal["all", "errors", "tools", "memory"]
_T = TypeVar("_T")
_P = ParamSpec("_P")


class _PryKwargs(TypedDict, total=False):
    json_fallback: bool | None
    show_budget: bool
    show_memory: bool
    show_tools: bool
    show_llm: bool
    max_rows: int
    redact_prompts: bool
    stream_override: list[str] | None
    filter_mode: FilterMode


_DEBUG_FLAG = "--debug"
_ACTIVE_PRY_LOCK = threading.RLock()
_ACTIVE_PRY: Pry | None = None

# Right-panel tab names — 7 focused debugging views
_RIGHT_VIEWS: tuple[str, ...] = (
    "event",  # detail of the stream-selected event
    "agents",  # agent lifecycle: runs, iterations, handoffs, spawns
    "tools",  # tool calls with full params, args, results
    "memory",  # memory + context + knowledge at selected stream point
    "guardrails",  # guardrail checks + last agent output
    "debug",  # debug breakpoints / checkpoints + current execution position
    "errors",  # all errors and warnings
)

# Backwards-compat alias
_View = Literal["event", "agents", "tools", "memory", "guardrails", "debug", "errors"]
_ALL_VIEWS: tuple[str, ...] = _RIGHT_VIEWS

_CAT_LLM = "llm"
_CAT_TOOL = "tool"
_CAT_MEMORY = "memory"
_CAT_BUDGET = "budget"
_CAT_AGENT = "agent"
_CAT_KNOWLEDGE = "knowledge"
_CAT_GUARDRAIL = "guardrail"
_CAT_CONTEXT = "context"
_CAT_OUTPUT = "output"

# Tab → letter shortcut
_RIGHT_KEY_MAP: dict[str, str] = {
    "e": "event",
    "a": "agents",
    "t": "tools",
    "m": "memory",
    "g": "guardrails",
    "d": "debug",
    "r": "errors",
}

# Tab → accent colour used for the right panel border
_TAB_COLOR: dict[str, str] = {
    "event": "cyan",
    "agents": "cyan",
    "tools": "yellow",
    "memory": "magenta",
    "guardrails": "red",
    "debug": "bold red",
    "errors": "red",
}

_ERROR_HOOKS = ("error", "fail", "exceeded", "blocked")
_TOOL_HOOKS = ("tool",)
_MEMORY_HOOKS = ("memory",)

_PANEL_ROWS_DEFAULT = 14  # fallback when console height unknown
_MAX_STR = 80
_MAX_STREAM_COLS = 66  # max visual chars for a stream-list line
_MAX_DETAIL_COLS = 58  # max visual chars for detail / hover content lines

# Hook names that block on pause/step (major lifecycle points)
_PAUSE_AT_HOOKS = (
    "llm.request",
    "tool.call",
    "agent.run",
    "agent.iteration",
    "debug.breakpoint",
)

# Group merging: when a *.end fires, update the matching *.start record in-place
# Maps end-hook substring → base key used as _pending_groups key
_END_TO_BASE: dict[str, str] = {
    "llm.request.end": "llm.request",
    "tool.call.end": "tool.call",
    "mcp.tool.call.end": "mcp.tool.call",
    "agent.run.end": "agent.run",
    "agent.iteration.end": "agent.iteration",
    "knowledge.search.end": "knowledge.search",
    "guardrail.check.end": "guardrail.check",
    "system_prompt.after_resolve": "system_prompt",
}
_START_TO_BASE: dict[str, str] = {
    "llm.request.start": "llm.request",
    "tool.call.start": "tool.call",
    "mcp.tool.call.start": "mcp.tool.call",
    "agent.run.start": "agent.run",
    "agent.iteration.start": "agent.iteration",
    "knowledge.search.start": "knowledge.search",
    "guardrail.check.start": "guardrail.check",
    "system_prompt.before_resolve": "system_prompt",
}

# Pre-compiled regex to strip Rich markup for visual-length measurement
_MARKUP_RE = re.compile(r"\[/?[^\[\]]*\]")

# Highlight styles — two variants so both panels look great
_HL_FOCUSED = "on steel_blue1"  # active panel selected row bg
_HL_UNFOCUSED = "on grey27"  # inactive panel selected row bg
_HL_AUTO = "on navy_blue"  # auto-follow last row bg


def _scrollbar_chars(total: int, visible: int, start: int) -> list[str]:
    """Single-character scrollbar column for a visible window.

    Args:
        total:   total number of items/lines in the full list.
        visible: number of rows shown (``_PANEL_ROWS``).
        start:   index of the first visible item.

    Returns:
        A list of ``visible`` Rich-markup strings (``│`` or ``█``).
    """
    if total <= visible:
        return ["[dim]│[/dim]"] * visible
    thumb_h = max(1, round(visible * visible / total))
    max_start = max(1, total - visible)
    frac = start / max_start
    thumb_top = round(frac * (visible - thumb_h))
    return [
        "[dim cyan]█[/dim cyan]" if thumb_top <= i < thumb_top + thumb_h else "[dim]│[/dim]"
        for i in range(visible)
    ]


def _plain_len(s: str) -> int:
    """Visual length of *s* with Rich markup tags removed."""
    return len(_MARKUP_RE.sub("", s))


def _truncate_markup(s: str, max_plain: int) -> str:
    """Truncate *s* to at most *max_plain* visible characters, preserving markup."""
    if _plain_len(s) <= max_plain:
        return s
    count = 0
    buf: list[str] = []
    i = 0
    while i < len(s):
        if s[i] == "[":
            j = s.find("]", i)
            if j != -1:
                buf.append(s[i : j + 1])
                i = j + 1
                continue
        if count >= max_plain - 1:
            break
        buf.append(s[i])
        count += 1
        i += 1
    return "".join(buf) + "…"


def _wrap_plain_lines(lines: list[str], width: int = _MAX_DETAIL_COLS) -> list[str]:
    """Word-wrap plain-text lines to *width* chars (no markup aware, safe for content)."""
    out: list[str] = []
    for line in lines:
        if len(line) <= width:
            out.append(line)
        else:
            wrapped = textwrap.wrap(
                line,
                width=width,
                subsequent_indent="  ",
                break_long_words=True,
                break_on_hyphens=False,
            )
            out.extend(wrapped or [line])
    return out


def _trunc(val: object, n: int = _MAX_STR) -> str:
    s = str(val)
    return s if len(s) <= n else s[: n - 1] + "…"


def _now() -> str:
    t = time.localtime()
    return f"{t.tm_hour:02d}:{t.tm_min:02d}:{t.tm_sec:02d}"


def _consume_debug_flag(argv: list[str] | None = None) -> bool:
    """Remove ``--debug`` from argv and report whether it was present."""
    target = sys.argv if argv is None else argv
    try:
        target.remove(_DEBUG_FLAG)
    except ValueError:
        return False
    return True


def _fmt_cost(raw: str) -> str:
    with contextlib.suppress(Exception):
        return f"[green]${float(raw):.4f}[/green]"
    return raw


def _fmt_dur(raw: str) -> str:
    with contextlib.suppress(Exception):
        return f"{float(raw):.2f}s"
    return raw


def _event_matches_filter(hook_name: str, mode: FilterMode) -> bool:
    if mode == "all":
        return True
    lname = hook_name.lower()
    if mode == "errors":
        return any(f in lname for f in _ERROR_HOOKS)
    if mode == "tools":
        return any(f in lname for f in _TOOL_HOOKS)
    if mode == "memory":
        return any(f in lname for f in _MEMORY_HOOKS)
    return True


@dataclass
class _EventRecord:
    ts: str
    hook: str
    badge: str  # "[color]icon label[/color]" — type badge, no timestamp
    brief: str  # short content snippet for the stream column
    category: str
    detail_lines: list[str] = field(default_factory=list)
    agent_name: str = ""
    model_name: str = ""
    full_content: str = ""  # full LLM response / system prompt / tool result

    @property
    def stream_line(self) -> str:
        """Compose the full display line from parts."""
        parts = [f"[dim]{self.ts}[/dim]  {self.badge}"]
        if self.model_name:
            parts.append(f"[dim]\\[{self.model_name}][/dim]")
        if self.agent_name:
            parts.append(f"[dim]{self.agent_name}[/dim]")
        if self.brief:
            parts.append(f"[dim]{self.brief}[/dim]")
        return "  ".join(parts)


def _is_error_record(r: _EventRecord) -> bool:
    """True when the record represents an error or warning."""
    h = r.hook.lower()
    return any(e in h for e in _ERROR_HOOKS)


class _DashboardRenderable:
    """Rich renderable — single render source, no concurrent-render race."""

    def __init__(self, ui: Pry) -> None:
        self._ui = ui

    def __rich_console__(self, console: Any, options: Any) -> Any:  # type: ignore[explicit-any]
        yield self._ui._render()


class _LiveStdout:
    """Proxy sys.stdout → event stream while the panel is active.

    print() calls become 'dev.log' events visible in the stream view instead
    of scrolling above the panel.
    """

    def __init__(self, live: Any, ui: Pry) -> None:  # type: ignore[explicit-any]
        self._live = live
        self._ui = ui
        self._buf = ""

    def write(self, s: str) -> int:
        if not s:
            return 0
        self._buf += s
        # Only flush when we have a complete "print" (trailing newline).
        # This batches multi-line prints into ONE event record.
        if self._buf.endswith("\n"):
            text = self._buf.rstrip("\n")
            self._buf = ""
            if text.strip():
                lines = [ln for ln in text.splitlines() if ln.strip()]
                if not lines:
                    return len(s)
                brief = _trunc(lines[0], 55)
                if len(lines) > 1:
                    brief += f" [dim]+{len(lines) - 1} lines[/dim]"
                if self._ui._started:
                    self._ui._events.append(
                        _EventRecord(
                            ts=_now(),
                            hook="dev.log",
                            badge="[dim white]📝 log[/dim white]",
                            brief=brief,
                            category=_CAT_OUTPUT,
                            detail_lines=[
                                "[dim white]📝 Developer log[/dim white]",
                                *[f"  {ln}" for ln in lines],
                            ],
                            full_content=text,
                        )
                    )
                else:
                    self._live.console.print(text, markup=False, highlight=False, end="\n")
        return len(s)

    def flush(self) -> None:
        if self._buf:
            text = self._buf
            self._buf = ""
            if self._ui._started and text.strip():
                self._ui._events.append(
                    _EventRecord(
                        ts=_now(),
                        hook="dev.log",
                        badge="[dim white]📝 log[/dim white]",
                        brief=_trunc(text, 80),
                        category=_CAT_OUTPUT,
                        detail_lines=["[dim white]📝 Developer log[/dim white]", f"  {text}"],
                        full_content=text,
                    )
                )
            elif text.strip():
                self._live.console.print(text, markup=False, highlight=False, end="")

    def isatty(self) -> bool:
        return False

    def fileno(self) -> int:
        raise io.UnsupportedOperation("fileno")

    @property
    def encoding(self) -> str:
        return "utf-8"

    @property
    def errors(self) -> str:
        return "replace"


class Pry:
    """Interactive debugger for syrin agents — like byebug, but for AI.

    Pry gives you a live two-panel TUI: the left column streams every agent
    event in real time; the right column lets you drill into agents, tools,
    memory, context snapshots, guardrails, and errors — each in its own focused
    tab.  Call ``pry.debugpoint("label")`` anywhere in your code to hard-pause
    execution and inspect the full agent state (context, memory, tools, budget),
    then press **p** to resume or **n** to step one hook at a time.

    **Attach and go**::

        pry = Pry()
        pry.attach(agent)
        agent.run("task")              # TUI stays open after run finishes
                                       # press q to exit

    **Breakpoints**::

        pry.attach(agent)
        agent.run("phase 1")
        pry.debugpoint("after phase 1")   # blocks here — inspect, then [p]
        agent.handoff(NextAgent, "phase 2")

    **Context manager** (recommended for scripts)::

        with Pry() as pry:
            pry.attach(agent)
            pry.run(agent.run, "task")  # background thread keeps TUI responsive
            pry.wait()                  # hold open until [q]

    **Tabs** — press the letter to jump to the tab:

    - **[e]** event detail of selected stream event
    - **[a]** agents — all runs, handoffs, spawns
    - **[t]** tools — full args + results for every call
    - **[m]** memory + context + knowledge (time-aware: state at selected event)
    - **[g]** guardrails + last agent output
    - **[d]** debug — breakpoints and execution position
    - **[r]** errors and warnings

    **Navigation**:

    - **↑/↓** scroll the focused panel
    - **←/→** switch focus between stream and right panel
    - **↵** drill into selected item (full detail)
    - **ESC** go back
    - **p** pause / resume   **n** step one hook   **q** quit

    Args:
        json_fallback: Force JSON lines mode regardless of TTY.
        show_budget: Include budget events. Default ``True``.
        show_memory: Include memory events. Default ``True``.
        show_tools: Include tool events. Default ``True``.
        show_llm: Include LLM call events. Default ``True``.
        max_rows: Scrollback buffer size. Default ``500``.
        redact_prompts: Replace content/args with ``[redacted]``.
        stream_override: Append JSON lines here instead of stdout (testing).
        filter_mode: ``"all"``, ``"errors"``, ``"tools"``, or ``"memory"``.
    """

    def __init__(
        self,
        json_fallback: bool | None = None,
        show_budget: bool = True,
        show_memory: bool = True,
        show_tools: bool = True,
        show_llm: bool = True,
        max_rows: int = 500,
        redact_prompts: bool = False,
        stream_override: list[str] | None = None,
        filter_mode: FilterMode = "all",
    ) -> None:
        if json_fallback is None:
            orig = getattr(sys, "__stdout__", sys.stdout)
            self.json_fallback = orig is None or not orig.isatty()
        else:
            self.json_fallback = json_fallback

        self.show_budget = show_budget
        self.show_memory = show_memory
        self.show_tools = show_tools
        self.show_llm = show_llm
        self.max_rows = max_rows
        self.redact_prompts = redact_prompts
        self.filter_mode: FilterMode = filter_mode

        self._agents: list[object] = []
        self._handlers: dict[str, object] = {}
        self._live: Any = None  # type: ignore[explicit-any]
        self._started = False
        self._paused = False
        self._stream_override = stream_override
        # Right-panel tab (independent of stream cursor)
        self._right_view: str = "event"
        self._right_view_idx: int = 0
        self._events: deque[_EventRecord] = deque(maxlen=max_rows)
        self._key_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._old_stdout: Any = None  # type: ignore[explicit-any]

        # Stream cursor navigation & mode
        self._cursor: int = -1  # -1 = auto-follow; 0+ = selected index
        self._mode: str = "browse"  # "browse" | "detail" | "right_detail" | "input"
        self._detail_scroll: int = 0

        # Right-panel cursor — mirrors stream cursor logic for the active tab's items
        self._right_cursor: int = -1  # -1 = auto-follow last item
        self._right_preview_scroll: int = 0
        self._right_detail_scroll: int = 0
        self._right_detail_rec: _EventRecord | None = None

        # Legacy per-view scroll dict (kept for compat, unused in new render path)
        self._scroll: dict[str, int] = {}

        # Pause gate — agents blocked in hooks until resumed
        self._pause_gate: threading.Event = threading.Event()
        self._pause_gate.set()  # set = running (not paused)
        # Step semaphore — each release() lets one blocked hook through
        self._step_mode: bool = False
        self._step_sem: threading.Semaphore = threading.Semaphore(0)

        # Group merging: track pending start records to update in-place on end
        self._pending_groups: dict[str, _EventRecord] = {}

        # Panel focus: "stream" = ↑↓ navigates stream; "right" = ↑↓ scrolls right panel
        self._focus: str = "stream"

        # Human-in-the-loop input channel
        self._input_prompt: str = ""
        self._input_buf: str = ""
        self._input_queue: queue.Queue[str] = queue.Queue()

        # Running stats
        self._total_cost: float = 0.0
        self._total_tokens: int = 0
        self._input_tokens: int = 0
        self._output_tokens: int = 0
        self._memory_count: int = 0
        self._error_count: int = 0
        self._budget_limit: float = 0.0

        # Context snapshot timeline
        self._ctx_timeline: list[str] = []

        # Track current model name (updated from agent.run.start)
        self._current_model: str = ""

        # Full context snapshots for "context at a point in time" viewing
        self._context_snapshots: list[tuple[str, dict[str, object]]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def from_debug_flag(
        cls,
        argv: list[str] | None = None,
        **kwargs: Unpack[_PryKwargs],
    ) -> Pry | None:
        """Create a Pry only when ``--debug`` is present, consuming the flag."""
        if not _consume_debug_flag(argv):
            return None
        return cls(**kwargs)

    def attach(self, agent: object) -> Pry:
        """Attach the UI to an agent and start the panel immediately.

        Args:
            agent: ``Agent`` instance to monitor.

        Returns:
            ``self`` — chainable: ``ui.attach(a1).attach(a2)``.
        """
        self._agents.append(agent)
        self._register_hooks(agent)
        # Read budget limit directly from the agent so the status bar shows it
        # from the very first render (budget.check events may never fire).
        with contextlib.suppress(Exception):
            _budget = getattr(agent, "_budget", None)
            if _budget is not None:
                _lim = float(getattr(_budget, "max_cost", 0) or 0)
                if _lim > 0:
                    self._budget_limit = _lim
        if not self._started:
            self.start()
        return self

    def detach(self) -> None:
        """Detach from all agents.

        In TUI mode the panel stays open for review until the user presses
        **q** — this lets scripts call ``detach()`` (or exit normally) without
        the terminal snapping back immediately.  In json_fallback / CI mode the
        panel is stopped immediately.
        """
        for agent in self._agents:
            self._deregister_hooks(agent)
        self._agents.clear()
        if self._started:
            if self._live is not None and not self._stop_event.is_set():
                # TUI is live — show completion marker and wait for q
                self._events.append(
                    _EventRecord(
                        ts=_now(),
                        hook="session.complete",
                        badge="[bold green]✓ done[/bold green]",
                        brief="press q to exit",
                        category=_CAT_OUTPUT,
                        detail_lines=[
                            "[bold green]✓ Session complete[/bold green]",
                            "  Press [bold]q[/bold] to exit",
                        ],
                    )
                )
                self._stop_event.wait()  # blocks until q → stop()
            self.stop()

    def wait(self) -> None:
        """Block until the user presses ``q``."""
        self._stop_event.wait()

    def start(self) -> None:
        """Start the panel. Called automatically by :meth:`attach`."""
        if self._started:
            return
        self._started = True
        if not self.json_fallback:
            self._claim_terminal()
            self._start_rich()

    def stop(self) -> None:
        """Stop the panel and restore the terminal.

        Order matters: we stop the Rich Live instance (killing its daemon
        refresh thread) BEFORE setting ``_stop_event``.  Any code waiting on
        ``_stop_event`` (atexit handler, ``detach()``, ``wait()``) will only
        unblock *after* Live is fully stopped — so Python's interpreter
        finalisation never races with a live Rich daemon thread.
        """
        self._started = False
        self._paused = False
        self._step_mode = False
        self._pause_gate.set()
        # Stop Live first so the daemon thread is dead before we unblock callers
        if self._live is not None:
            with contextlib.suppress(Exception):
                live = self._live
                self._live = None
                live.stop()
        if self._old_stdout is not None:
            sys.stdout = self._old_stdout
            self._old_stdout = None
        self._release_terminal()
        # Set last — callers waiting on this event can now safely proceed
        self._stop_event.set()

    def set_filter(self, mode: FilterMode) -> None:
        """Set the event filter applied in the stream view."""
        self.filter_mode = mode

    def pause(self) -> None:
        """Pause agent execution at the next major hook point.

        The agent thread blocks until :meth:`resume` or a step via :meth:`step`.
        """
        self._paused = True
        self._step_mode = False
        self._pause_gate.clear()

    def resume(self) -> None:
        """Resume agent execution from a pause."""
        self._paused = False
        self._step_mode = False
        self._pause_gate.set()

    def step(self) -> None:
        """Advance one hook step while paused (like a debugger's "next").

        Unblocks the agent thread for exactly one major hook, then re-blocks it.
        Call programmatically or press **n** in the TUI when paused.
        """
        if not self._paused:
            return
        self._step_mode = True
        # Unblock the agent thread — it is waiting on _pause_gate.wait().
        # The hook handler will re-clear the gate after the single step.
        self._pause_gate.set()

    def debugpoint(self, label: str = "") -> None:
        """Hard-pause execution here and capture full agent state — like pry.binding.

        Call this anywhere between or during agent calls::

            pry.debugpoint("before handoff")
            # ↑ execution stops immediately here.
            # Navigate all tabs: context, memory, tools, budget, errors.
            # Press [p] to resume, [n] to step one hook.

        Unlike pause(), debugpoint() blocks the *calling thread* right here —
        not at the next hook. Full state from every attached agent is captured
        and shown in the [d] debug tab.
        """
        state_lines, state_full = self._capture_agent_states()
        detail: list[str] = [
            "[bold red]🔴 Debug point[/bold red]",
            f"  label: [bold]{label}[/bold]" if label else "",
            "",
            "[dim]Navigate tabs to inspect agent state:[/dim]",
            "  [e] event detail   [a] agents   [t] tools",
            "  [m] memory         [g] guardrails  [r] errors",
            "  Press [bold]p[/bold] to resume  •  [bold]n[/bold] to step  •  [bold]q[/bold] to quit",
        ]
        detail.extend(state_lines)
        detail = [item for item in detail if item is not None]

        self._events.append(
            _EventRecord(
                ts=_now(),
                hook="debug.breakpoint",
                badge="[bold red]🔴 debugpoint[/bold red]",
                brief=label or "debugpoint",
                category=_CAT_OUTPUT,
                detail_lines=detail,
                full_content=state_full,
            )
        )

        # Block the calling thread immediately — not at the next hook.
        self._paused = True
        self._step_mode = False
        self._pause_gate.clear()
        self._pause_gate.wait()  # unblocked by resume() or step()

    # Keep checkpoint() as an alias so existing code doesn't break silently.
    def checkpoint(self, label: str = "") -> None:
        """Alias for :meth:`debugpoint`. Prefer debugpoint() in new code."""
        self.debugpoint(label)

    def _capture_agent_states(self) -> tuple[list[str], str]:
        """Snapshot state from every attached agent for the debug panel.

        Returns (rich_markup_lines, plain_full_content).
        """
        lines: list[str] = []
        full_parts: list[str] = []

        # Collect all agents: directly attached + their spawned children
        all_agents: list[object] = []
        for agent in self._agents:
            all_agents.append(agent)
            for child in getattr(agent, "_spawned_children", []):
                all_agents.append(child)

        for agent in all_agents:
            name = getattr(agent, "name", getattr(agent, "_agent_name", type(agent).__name__))
            lines.append("")
            lines.append(
                f"[dim]─── agent: [bold cyan]{name}[/bold cyan] ──────────────────────────────[/dim]"
            )
            full_parts.append(f"\n=== agent: {name} ===\n")

            # ── Model ─────────────────────────────────────────────────────
            mc = getattr(agent, "model_config", None)
            if mc is not None:
                model_name = getattr(mc, "model_name", getattr(mc, "name", str(mc)))
                lines.append(f"  model:      [bold blue]{model_name}[/bold blue]")
                full_parts.append(f"model: {model_name}")

            # ── Budget ────────────────────────────────────────────────────
            with contextlib.suppress(Exception):
                bs = getattr(agent, "budget_state", None)
                if bs is not None:
                    pct_color = (
                        "green"
                        if bs.percent_used < 50
                        else ("yellow" if bs.percent_used < 80 else "red")
                    )
                    lines.append(
                        f"  budget:     [bold]${bs.spent:.4f}[/bold] / ${bs.limit:.4f}"
                        f"  [{pct_color}]{bs.percent_used:.1f}%[/{pct_color}]  "
                        f"(${bs.remaining:.4f} remaining)"
                    )
                    full_parts.append(
                        f"budget: ${bs.spent:.4f} spent / ${bs.limit:.4f} limit  "
                        f"({bs.percent_used:.1f}% used, ${bs.remaining:.4f} remaining)"
                    )

            # ── Context stats ─────────────────────────────────────────────
            with contextlib.suppress(Exception):
                cs = getattr(agent, "context_stats", None)
                if cs is not None:
                    tok = getattr(cs, "total_tokens", None) or getattr(cs, "tokens_used", None)
                    max_tok = getattr(cs, "max_tokens", None)
                    compact_count = getattr(cs, "compact_count", 0)
                    if tok is not None:
                        tok_line = f"  context:    [bold]{tok}[/bold] tok"
                        if max_tok:
                            tok_line += f" / {max_tok}"
                        if compact_count:
                            tok_line += f"  [yellow]{compact_count}x compacted[/yellow]"
                        lines.append(tok_line)
                        full_parts.append(
                            f"context: {tok} tokens" + (f" / {max_tok} max" if max_tok else "")
                        )

            # ── Tools ─────────────────────────────────────────────────────
            with contextlib.suppress(Exception):
                tools = getattr(agent, "tools", None) or []
                if tools:
                    names = [getattr(t, "name", str(t)) for t in tools]
                    lines.append(f"  tools ({len(tools)}):  [yellow]{', '.join(names)}[/yellow]")
                    full_parts.append(f"\ntools ({len(tools)}):")
                    for t in tools:
                        tname = getattr(t, "name", "?")
                        tdesc = getattr(t, "description", "")
                        full_parts.append(f"  • {tname}: {tdesc}")

            # ── Memory ────────────────────────────────────────────────────
            with contextlib.suppress(Exception):
                mem = getattr(agent, "memory", None)
                if mem is not None:
                    backend = getattr(mem, "backend", "—")
                    top_k = getattr(mem, "top_k", "—")
                    restrict = getattr(mem, "types", None)
                    auto_store = getattr(mem, "auto_store", False)
                    scope = getattr(mem, "scope", "—")
                    types_str = ", ".join(str(t) for t in restrict) if restrict else "all"
                    lines.append(
                        f"  memory:     backend=[magenta]{backend}[/magenta]"
                        f"  top_k={top_k}  types={types_str}"
                        f"  scope={scope}" + ("  [dim]auto_store[/dim]" if auto_store else "")
                    )
                    full_parts.append(
                        f"\nmemory: backend={backend}  top_k={top_k}  "
                        f"types={types_str}  scope={scope}"
                        + (f"  auto_store={auto_store}" if auto_store else "")
                    )
                    # Remembered items count (if store is accessible)
                    with contextlib.suppress(Exception):
                        store = getattr(mem, "_store", None) or getattr(mem, "store", None)
                        if store is not None:
                            count_fn = getattr(store, "count", None)
                            if callable(count_fn):
                                count = count_fn()
                                lines.append(f"  mem items:  [magenta]{count}[/magenta] stored")
                                full_parts.append(f"  stored items: {count}")

            # ── Rate limit ────────────────────────────────────────────────
            with contextlib.suppress(Exception):
                rl = getattr(agent, "rate_limit", None)
                if rl is not None:
                    rpm = getattr(rl, "requests_per_minute", None)
                    tpm = getattr(rl, "tokens_per_minute", None)
                    if rpm or tpm:
                        parts_rl = []
                        if rpm:
                            parts_rl.append(f"rpm={rpm}")
                        if tpm:
                            parts_rl.append(f"tpm={tpm}")
                        lines.append(f"  rate limit: [dim]{' '.join(parts_rl)}[/dim]")
                        full_parts.append(f"rate limit: {' '.join(parts_rl)}")

            # ── Context snapshot (live — from last LLM call) ──────────────
            with contextlib.suppress(Exception):
                ctx_mgr = getattr(agent, "_context", None)
                snap = getattr(ctx_mgr, "_last_snapshot", None)
                if snap is not None:
                    snap_lines, snap_full = self._fmt_context_snapshot(snap)
                    lines.extend(snap_lines)
                    if snap_full:
                        full_parts.append("")
                        full_parts.append(snap_full)

        return lines, "\n".join(full_parts)

    def wait_if_paused(self) -> None:
        """Block the calling thread until the UI is resumed.

        Call this inside agent hooks to implement true agent pause::

            @agent.events.on("llm.request.start")
            def gate(hook, ctx):
                ui.wait_if_paused()
        """
        self._pause_gate.wait()

    def run(self, fn: Callable[_P, _T], *args: _P.args, **kwargs: _P.kwargs) -> threading.Thread:
        """Run *fn* in a background thread so the TUI key loop stays responsive.

        This is the recommended way to execute agents inside a debug session —
        calling ``agent.run()`` directly on the main thread blocks key input.

        Args:
            fn: Callable to run (e.g. ``agent.run`` or a ``@task`` method).
            *args: Positional arguments forwarded to *fn*.
            **kwargs: Keyword arguments forwarded to *fn*.

        Returns:
            The background :class:`threading.Thread` (already started).

        Example::

            with Pry() as ui:
                ui.attach(agent)
                t = ui.run(agent.run, "Hello")
                t.join()
                ui.wait()
        """
        t = threading.Thread(target=fn, args=args, kwargs=kwargs, daemon=True)
        t.start()
        return t

    def get_input(self, prompt: str = "") -> str:
        """Show an input prompt in the TUI and return the user's typed response.

        Switches the UI to input mode; the user types and presses Enter.
        Blocks until Enter is pressed. Useful for human-in-the-loop agents::

            answer = ui.get_input("Approve action? (y/n): ")
        """
        self._input_prompt = prompt
        self._input_buf = ""
        self._mode = "input"
        result = self._input_queue.get()  # blocks until Enter pressed
        self._mode = "browse"
        return result

    def __enter__(self) -> Pry:
        return self

    def __exit__(self, *_: object) -> None:
        self.detach()

    def _claim_terminal(self) -> None:
        """Ensure only one live Pry owns the terminal at a time."""
        global _ACTIVE_PRY
        with _ACTIVE_PRY_LOCK:
            other = _ACTIVE_PRY
            if other is self:
                return
            if other is not None:
                other._shutdown_for_replacement()
            _ACTIVE_PRY = self

    def _release_terminal(self) -> None:
        global _ACTIVE_PRY
        with _ACTIVE_PRY_LOCK:
            if _ACTIVE_PRY is self:
                _ACTIVE_PRY = None

    def _shutdown_for_replacement(self) -> None:
        """Drop hooks and stop immediately when another Pry takes over."""
        for agent in list(self._agents):
            self._deregister_hooks(agent)
        self._agents.clear()
        self.stop()

    # ------------------------------------------------------------------
    # Internal — hook registration
    # ------------------------------------------------------------------

    def _register_hooks(self, agent: object) -> None:
        if not hasattr(agent, "events"):
            return
        agent_name = getattr(agent, "name", getattr(agent, "_agent_name", ""))

        def _all_handler(hook: object, ctx: object) -> None:
            hook_str = str(hook)
            # ── True pause / step gate ────────────────────────────────────
            # Block the agent thread at major lifecycle hooks so "pause" is
            # a real execution stop, not just a UI filter.
            if not self._pause_gate.is_set() and any(kw in hook_str for kw in _PAUSE_AT_HOOKS):
                # Block until resume() or step() sets the gate
                self._pause_gate.wait()
                # If step() woke us, re-clear the gate immediately so
                # the agent pauses again after this single hook.
                if self._step_mode:
                    self._step_mode = False
                    self._pause_gate.clear()
            if self._started:
                self._handle_event(hook_str, ctx, agent_name)

        key = str(id(agent))
        self._handlers[key] = _all_handler
        with contextlib.suppress(Exception):
            agent.events.on_all(_all_handler)

    def _deregister_hooks(self, agent: object) -> None:
        key = str(id(agent))
        self._handlers.pop(key, None)

    # ------------------------------------------------------------------
    # Internal — event dispatch
    # ------------------------------------------------------------------

    def _handle_event(self, hook_name: str, ctx: object, _agent_hint: str = "") -> None:
        # In json_fallback (CI) mode, honour pause as an event filter too.
        if self._paused and self.json_fallback:
            return
        if not _event_matches_filter(hook_name, self.filter_mode):
            return

        if self.json_fallback:
            self._emit_json(hook_name, ctx)
            return

        self._update_stats(hook_name, ctx)

        # ── Group merging ────────────────────────────────────────────────────
        # Check if this is an "end" hook that should update an existing record
        h_lower = hook_name.lower()
        end_base = next((v for k, v in _END_TO_BASE.items() if k in h_lower), None)
        if end_base and end_base in self._pending_groups:
            pending = self._pending_groups.pop(end_base)
            end_rec = self._build_record(hook_name, ctx)
            if end_rec is not None:
                # Update the pending start-record in place (it's already in _events)
                pending.hook = hook_name
                pending.badge = end_rec.badge
                pending.brief = end_rec.brief
                pending.detail_lines = end_rec.detail_lines
                pending.full_content = end_rec.full_content
                if end_rec.model_name:
                    pending.model_name = end_rec.model_name
            return  # do NOT append a new record

        rec = self._build_record(hook_name, ctx)
        if rec is None:
            return
        if not rec.agent_name:
            rec.agent_name = self._g(ctx, "agent_name", _agent_hint)
        if not rec.model_name:
            rec.model_name = self._g(ctx, "model", "")
        self._events.append(rec)

        # If this is a "start" hook, register it as a pending group
        start_base = next((v for k, v in _START_TO_BASE.items() if k in h_lower), None)
        if start_base:
            self._pending_groups[start_base] = rec

    def _emit_json(self, hook_name: str, ctx: object) -> None:
        data: dict[str, object] = {"event": hook_name, "level": "INFO"}
        if isinstance(ctx, dict):
            for k, v in ctx.items():
                data[k] = v
        line = json.dumps(data, default=str)
        if self._stream_override is not None:
            self._stream_override.append(line)
        else:
            out = self._old_stdout if self._old_stdout is not None else sys.stdout
            out.write(line + "\n")
            out.flush()

    def _update_stats(self, hook_name: str, ctx: object) -> None:
        h = hook_name.lower()
        g = self._g

        # Model name only available on agent.run.start
        if "agent.run.start" in h:
            model = g(ctx, "model", "")
            if model:
                self._current_model = model

        if "llm.request.end" in h:
            with contextlib.suppress(Exception):
                tok = int(g(ctx, "tokens", "0") or "0")
                if tok > 0:
                    self._total_tokens += tok
            with contextlib.suppress(Exception):
                in_tok = int(g(ctx, "input_tokens", "0") or "0")
                if in_tok > 0:
                    self._input_tokens += in_tok
            with contextlib.suppress(Exception):
                out_tok = int(g(ctx, "output_tokens", "0") or "0")
                if out_tok > 0:
                    self._output_tokens += out_tok
            with contextlib.suppress(Exception):
                cost_val = float(g(ctx, "cost", "0") or "0")
                if cost_val > 0:
                    self._total_cost += cost_val

        if "agent.run.end" in h:
            with contextlib.suppress(Exception):
                tok = int(g(ctx, "tokens", "0") or "0")
                if tok > 0:
                    self._total_tokens = max(self._total_tokens, tok)
            with contextlib.suppress(Exception):
                in_tok = int(g(ctx, "input_tokens", "0") or "0")
                if in_tok > 0:
                    self._input_tokens = max(self._input_tokens, in_tok)
            with contextlib.suppress(Exception):
                out_tok = int(g(ctx, "output_tokens", "0") or "0")
                if out_tok > 0:
                    self._output_tokens = max(self._output_tokens, out_tok)
            with contextlib.suppress(Exception):
                cost_val = float(g(ctx, "cost", "0") or "0")
                if cost_val > 0:
                    self._total_cost = max(self._total_cost, cost_val)

        # budget.threshold uses current_value / limit_value (not used / limit)
        if "budget.threshold" in h:
            with contextlib.suppress(Exception):
                v = float(g(ctx, "current_value", g(ctx, "used", "")) or "0")
                if v > 0:
                    self._total_cost = v
            with contextlib.suppress(Exception):
                lim = float(g(ctx, "limit_value", g(ctx, "limit", "")) or "0")
                if lim > 0:
                    self._budget_limit = lim
        elif "budget.check" in h:
            with contextlib.suppress(Exception):
                used = float(g(ctx, "used", g(ctx, "spent", "0")) or "0")
                if used >= 0:
                    self._total_cost = used
            with contextlib.suppress(Exception):
                remaining = float(g(ctx, "remaining", "0") or "0")
                total = float(g(ctx, "total", "0") or "0")
                if total > 0:
                    self._budget_limit = total
                elif remaining >= 0 and self._total_cost >= 0:
                    self._budget_limit = self._total_cost + remaining
        elif "budget.exceeded" in h:
            with contextlib.suppress(Exception):
                used = float(g(ctx, "used", g(ctx, "spent", "0")) or "0")
                limit = float(g(ctx, "limit", "0") or "0")
                if used > 0:
                    self._total_cost = used
                if limit > 0:
                    self._budget_limit = limit

        if "memory.store" in h:
            self._memory_count += 1
        elif "memory.forget" in h:
            self._memory_count = max(0, self._memory_count - 1)

        if any(e in h for e in _ERROR_HOOKS):
            self._error_count += 1

        ts = _now()
        if "context.compress" in h or "context.compact" in h:
            # context.compact uses tokens_before/tokens_after
            before = g(ctx, "tokens_before", g(ctx, "initial_tokens", g(ctx, "before_tokens", "?")))
            after = g(ctx, "tokens_after", g(ctx, "final_tokens", g(ctx, "after_tokens", "?")))
            ratio = g(ctx, "compression_ratio", g(ctx, "ratio", ""))
            entry = f"[dim]{ts}[/dim]  [steel_blue1]◑ compress[/steel_blue1]  {before}→{after} tok"
            if ratio:
                entry += f"  ratio={ratio}"
            self._ctx_timeline.append(entry)
        elif "context.threshold" in h:
            pct = g(ctx, "percent", g(ctx, "usage", ""))
            tokens = g(ctx, "tokens", "")
            self._ctx_timeline.append(
                f"[dim]{ts}[/dim]  [yellow]▲ threshold {pct}%[/yellow]"
                + (f"  {tokens} tok" if tokens else "")
            )
        elif "context.snapshot" in h:
            # Payload: {"snapshot": snap.to_dict(), "utilization_pct": float}
            snap_data = ctx.get("snapshot", {}) if isinstance(ctx, dict) else {}
            sd = snap_data if isinstance(snap_data, dict) else {}
            tokens = sd.get("total_tokens", "")
            snap_n = len(self._context_snapshots) + 1
            if sd:
                self._context_snapshots.append((ts, sd))
            self._ctx_timeline.append(
                f"[dim]{ts}[/dim]  [steel_blue1]📷 snapshot #{snap_n}[/steel_blue1]"
                + (f"  {tokens} tok" if tokens else "")
            )
        elif "context.offload" in h:
            offload_val = g(ctx, "offloaded_tokens", g(ctx, "turns", ""))
            self._ctx_timeline.append(
                f"[dim]{ts}[/dim]  [steel_blue1]⬆ offload[/steel_blue1]  {offload_val}"
            )
        elif "checkpoint.save" in h:
            cid = g(ctx, "checkpoint_id", g(ctx, "path", ""))
            self._ctx_timeline.append(
                f"[dim]{ts}[/dim]  [steel_blue1]💾 checkpoint {_trunc(cid, 30)}[/steel_blue1]"
            )

        if len(self._ctx_timeline) > 100:
            self._ctx_timeline = self._ctx_timeline[-100:]

    # ------------------------------------------------------------------
    # Internal — hook_name → _EventRecord
    # ------------------------------------------------------------------

    def _g(self, ctx: object, key: str, default: str = "") -> str:
        if isinstance(ctx, dict):
            return str(ctx.get(key, default))
        return str(getattr(ctx, key, default))

    def _raw(self, ctx: object, key: str) -> object:
        """Return the raw (non-stringified) value from hook context."""
        if isinstance(ctx, dict):
            return ctx.get(key)
        return getattr(ctx, key, None)

    @staticmethod
    def _fmt_context_snapshot(snap: object) -> tuple[list[str], str]:
        """Render a ContextSnapshot into (detail_lines, full_content).

        Returns rich markup lines for the detail panel and plain text for
        the scrollable full_content section (raw messages).
        """
        # Try importing — may not always be present
        try:
            from syrin.context.snapshot import ContextSnapshot
        except ImportError:
            return [f"  context: {snap}"], ""

        if not isinstance(snap, ContextSnapshot):
            return [f"  context: {snap}"], ""

        lines: list[str] = []
        lines.append("")
        lines.append("[dim]─── context snapshot ───────────────────────────────────────[/dim]")

        # Token summary
        tok = snap.total_tokens
        maxt = snap.max_tokens
        pct = f"{snap.utilization_pct:.1f}%" if snap.utilization_pct else "—"
        risk_color = {"low": "green", "medium": "yellow", "high": "red"}.get(
            snap.context_rot_risk, "white"
        )
        lines.append(f"  tokens:     [bold]{tok}[/bold] / {maxt}  ({pct} utilized)")
        lines.append(f"  available:  {snap.tokens_available}")
        lines.append(f"  rot risk:   [{risk_color}]{snap.context_rot_risk}[/{risk_color}]")
        if snap.compacted:
            lines.append(f"  compacted:  [yellow]yes[/yellow] ({snap.compact_method or 'unknown'})")
        if snap.context_mode and snap.context_mode != "full":
            lines.append(
                f"  ctx mode:   [cyan]{snap.context_mode}[/cyan]"
                + (
                    f"  (dropped {snap.context_mode_dropped_count})"
                    if snap.context_mode_dropped_count
                    else ""
                )
            )

        # Token breakdown
        bd = snap.breakdown
        if bd.total_tokens > 0:
            lines.append("")
            lines.append("  [dim]breakdown:[/dim]")
            if bd.system_tokens:
                lines.append(f"    system:    {bd.system_tokens} tok")
            if bd.tools_tokens:
                lines.append(f"    tools:     {bd.tools_tokens} tok")
            if bd.memory_tokens:
                lines.append(f"    memory:    {bd.memory_tokens} tok")
            if bd.messages_tokens:
                lines.append(f"    messages:  {bd.messages_tokens} tok")
            if bd.injected_tokens:
                lines.append(f"    injected:  {bd.injected_tokens} tok")

        # Message previews
        if snap.message_preview:
            lines.append("")
            lines.append(f"  [dim]messages ({len(snap.message_preview)}):[/dim]")
            role_color = {
                "system": "dim yellow",
                "user": "cyan",
                "assistant": "green",
                "tool": "magenta",
            }
            for i, mp in enumerate(snap.message_preview, 1):
                rc = role_color.get(mp.role, "white")
                tok_str = f"[dim]{mp.token_count} tok[/dim]" if mp.token_count else ""
                src_str = f"[dim]{mp.source}[/dim]" if mp.source else ""
                lines.append(f"  [{i}] [{rc}]{mp.role:<12}[/{rc}] {tok_str}  {src_str}")
                # Show full snippet (not truncated)
                if mp.content_snippet:
                    for chunk in mp.content_snippet.splitlines()[:6]:
                        lines.append(f"      [dim]{chunk}[/dim]")
                    if len(mp.content_snippet.splitlines()) > 6:
                        lines.append(
                            f"      [dim]… ({len(mp.content_snippet.splitlines()) - 6} more lines in raw messages)[/dim]"
                        )

        # Why included
        if snap.why_included:
            lines.append("")
            lines.append("  [dim]why included:[/dim]")
            for reason in snap.why_included:
                lines.append(f"    • {reason}")

        # Build full_content from raw_messages
        full_content = ""
        if snap.raw_messages:
            parts: list[str] = ["=== raw messages ===", ""]
            for i, msg in enumerate(snap.raw_messages, 1):
                role = str(msg.get("role", "?")) if isinstance(msg, dict) else "?"
                content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
                parts.append(f"[{i}] {role.upper()}")
                if isinstance(content, str):
                    parts.extend(content.splitlines())
                elif isinstance(content, list):
                    # Multi-part content (e.g. tool calls)
                    for part in content:
                        if isinstance(part, dict):
                            ptype = part.get("type", "")
                            ptext = part.get("text", part.get("content", str(part)))
                            parts.append(f"  ({ptype}) {ptext}")
                        else:
                            parts.append(f"  {part}")
                parts.append("")
            full_content = "\n".join(parts)

        return lines, full_content

    def _build_record(self, hook_name: str, ctx: object) -> _EventRecord | None:  # noqa: PLR0911, PLR0912, PLR0915
        h = hook_name.lower()
        ts = _now()
        g = self._g

        # ── LLM ─────────────────────────────────────────────────────────────
        if "llm.request.start" in h and self.show_llm:
            it = g(ctx, "iteration", "?")
            model = g(ctx, "model", "")
            return _EventRecord(
                ts,
                hook_name,
                badge=f"[blue]→ LLM #{it}[/blue]",
                brief=model,
                category=_CAT_LLM,
                detail_lines=[
                    f"[blue]→ LLM request #{it}[/blue]",
                    f"  model: {model or '—'}",
                ],
            )

        if "llm.request.end" in h and self.show_llm:
            it = g(ctx, "iteration", "?")
            tokens = g(ctx, "tokens", "")
            cost = _fmt_cost(g(ctx, "cost", ""))
            content = g(ctx, "content", "")
            if self.redact_prompts:
                content = "[redacted]"
            brief_parts = []
            if tokens:
                brief_parts.append(f"tok={tokens}")
            if cost:
                brief_parts.append(cost)
            if content:
                brief_parts.append(_trunc(content, 40))
            return _EventRecord(
                ts,
                hook_name,
                badge=f"[bold green]✓ LLM #{it}[/bold green]",
                brief="  ".join(brief_parts),
                category=_CAT_LLM,
                detail_lines=[
                    f"[bold green]✓ LLM response #{it}[/bold green]",
                    f"  tokens:  {tokens or '—'}",
                    f"  cost:    {cost or '—'}",
                    f"  preview: {_trunc(content, 120)}",
                ],
                full_content=content,
            )

        if "llm.retry" in h and self.show_llm:
            attempt = g(ctx, "attempt", "?")
            reason = g(ctx, "reason", g(ctx, "error", ""))
            return _EventRecord(
                ts,
                hook_name,
                badge=f"[yellow]↺ LLM retry #{attempt}[/yellow]",
                brief=_trunc(reason, 55),
                category=_CAT_LLM,
                detail_lines=[
                    f"[yellow]↺ LLM retry attempt {attempt}[/yellow]",
                    f"  reason: {_trunc(reason, 120)}",
                ],
            )

        if "llm.fallback" in h and self.show_llm:
            from_m = g(ctx, "from_model", g(ctx, "model", ""))
            to_m = g(ctx, "to_model", g(ctx, "fallback_model", ""))
            reason = g(ctx, "reason", "")
            return _EventRecord(
                ts,
                hook_name,
                badge="[bold yellow]⇒ FALLBACK[/bold yellow]",
                brief=f"{from_m} → {to_m}",
                category=_CAT_LLM,
                detail_lines=[
                    "[bold yellow]⇒ Model fallback[/bold yellow]",
                    f"  from:   {from_m or '—'}",
                    f"  to:     {to_m or '—'}",
                    f"  reason: {reason or '—'}",
                ],
            )

        if ("model.switch" in h or "model.switched" in h) and self.show_llm:
            from_m = g(ctx, "from_model", g(ctx, "previous", ""))
            to_m = g(ctx, "to_model", g(ctx, "model", ""))
            return _EventRecord(
                ts,
                hook_name,
                badge="[bold cyan]⇄ MODEL SWITCH[/bold cyan]",
                brief=f"{from_m} → {to_m}",
                category=_CAT_LLM,
                detail_lines=[
                    "[bold cyan]⇄ Model switched[/bold cyan]",
                    f"  from: {from_m or '—'}",
                    f"  to:   {to_m or '—'}",
                ],
            )

        if "routing.decision" in h:
            model = g(ctx, "model", g(ctx, "selected", ""))
            reason = g(ctx, "routing_reason", g(ctx, "reason", ""))
            return _EventRecord(
                ts,
                hook_name,
                badge=f"[cyan]⇢ route → {model}[/cyan]",
                brief=_trunc(reason, 50),
                category=_CAT_LLM,
                detail_lines=[
                    "[cyan]⇢ Routing decision[/cyan]",
                    f"  model:  {model or '—'}",
                    f"  reason: {_trunc(reason, 100)}",
                ],
            )

        # ── Tools ────────────────────────────────────────────────────────────
        if ("tool.call.start" in h or "mcp.tool.call.start" in h) and self.show_tools:
            # hook emits 'name' (not 'tool_name'); 'arguments' (not 'args')
            name = g(ctx, "name", g(ctx, "tool_name", "?"))
            args = g(ctx, "arguments", g(ctx, "args", ""))
            if self.redact_prompts:
                args = "[redacted]"
            prefix = "[dim]MCP [/dim]" if "mcp" in h else ""
            # Pretty-print JSON args if possible
            try:
                import json as _json

                parsed = _json.loads(args)
                pretty_args = _json.dumps(parsed, indent=2)
            except Exception:
                pretty_args = args
            return _EventRecord(
                ts,
                hook_name,
                badge=f"[bold yellow]{prefix}⚙ {name}[/bold yellow]",
                brief=_trunc(args, 55),
                category=_CAT_TOOL,
                detail_lines=[
                    f"[bold yellow]{prefix}⚙ Tool: {name}[/bold yellow]",
                ],
                full_content=f"args:\n{pretty_args}",
            )

        if ("tool.call.end" in h or "mcp.tool.call.end" in h) and self.show_tools:
            name = g(ctx, "name", g(ctx, "tool_name", "?"))
            result = g(ctx, "result", g(ctx, "output", ""))
            dur = _fmt_dur(g(ctx, "duration", ""))
            brief_parts = [dur] if dur else []
            brief_parts.append(_trunc(result, 40))
            return _EventRecord(
                ts,
                hook_name,
                badge=f"[bold green]✓ {name}[/bold green]",
                brief="  ".join(p for p in brief_parts if p),
                category=_CAT_TOOL,
                detail_lines=[
                    f"[bold green]✓ Tool result: {name}[/bold green]",
                    f"  duration: {dur or '—'}",
                    f"  result:   {_trunc(result, 120)}",
                ],
                full_content=result,
            )

        if ("tool.error" in h or ("tool" in h and "error" in h)) and self.show_tools:
            name = g(ctx, "name", g(ctx, "tool_name", "?"))
            err = g(ctx, "error", g(ctx, "message", ""))
            return _EventRecord(
                ts,
                hook_name,
                badge=f"[bold red]✗ {name}[/bold red]",
                brief=_trunc(err, 55),
                category=_CAT_TOOL,
                detail_lines=[
                    f"[bold red]✗ Tool error: {name}[/bold red]",
                ],
                full_content=err,
            )

        # ── Memory ───────────────────────────────────────────────────────────
        if "memory.store" in h and self.show_memory:
            mtype = g(ctx, "memory_type", g(ctx, "type", ""))
            mid = g(ctx, "memory_id", g(ctx, "key", ""))
            imp = g(ctx, "importance", "")
            brief_parts = [f"type={mtype}"] if mtype else []
            if imp:
                brief_parts.append(f"imp={imp}")
            brief_parts.append(_trunc(mid, 35))
            return _EventRecord(
                ts,
                hook_name,
                badge="[magenta]↪ store[/magenta]",
                brief="  ".join(p for p in brief_parts if p),
                category=_CAT_MEMORY,
                detail_lines=[
                    "[magenta]↪ Memory store[/magenta]",
                    f"  id:         {_trunc(mid, 80)}",
                    f"  type:       {mtype or '—'}",
                    f"  importance: {imp or '—'}",
                ],
            )

        if "memory.recall" in h and self.show_memory:
            query = g(ctx, "query", g(ctx, "key", ""))
            mtype = g(ctx, "memory_type", g(ctx, "type", ""))
            hits = g(ctx, "results_count", g(ctx, "count", g(ctx, "hits", "")))
            return _EventRecord(
                ts,
                hook_name,
                badge="[magenta]↩ recall[/magenta]",
                brief=f"type={mtype}  hits={hits}  {_trunc(query, 30)}",
                category=_CAT_MEMORY,
                detail_lines=[
                    "[magenta]↩ Memory recall[/magenta]",
                    f"  type:   {mtype or '—'}",
                    f"  hits:   {hits or '—'}",
                    f"  query:  {_trunc(query, 120)}",
                ],
            )

        if "memory.forget" in h and self.show_memory:
            mid = g(ctx, "memory_id", g(ctx, "key", ""))
            count = g(ctx, "deleted_count", "")
            return _EventRecord(
                ts,
                hook_name,
                badge="[magenta]✗ forget[/magenta]",
                brief=_trunc(mid, 55) + (f"  deleted={count}" if count else ""),
                category=_CAT_MEMORY,
                detail_lines=[
                    "[magenta]✗ Memory forget[/magenta]",
                    f"  id:      {_trunc(mid, 120)}",
                    f"  deleted: {count or '—'}",
                ],
            )

        if "memory.consolidate" in h and self.show_memory:
            # hook emits: memories_consolidated (not memories_merged)
            count = g(ctx, "memories_consolidated", g(ctx, "count", g(ctx, "memories_merged", "")))
            return _EventRecord(
                ts,
                hook_name,
                badge="[magenta]⊕ consolidate[/magenta]",
                brief=f"merged={count}",
                category=_CAT_MEMORY,
                detail_lines=[
                    "[magenta]⊕ Memory consolidate[/magenta]",
                    f"  merged: {count or '—'}",
                ],
            )

        if "memory.extract" in h and self.show_memory:
            count = g(ctx, "count", g(ctx, "extracted", ""))
            return _EventRecord(
                ts,
                hook_name,
                badge="[magenta]◈ extract[/magenta]",
                brief=f"count={count}",
                category=_CAT_MEMORY,
                detail_lines=[
                    "[magenta]◈ Memory extract[/magenta]",
                    f"  count: {count or '—'}",
                ],
            )

        # ── Budget & Rate Limits ─────────────────────────────────────────────
        if "budget.exceeded" in h and self.show_budget:
            used = g(ctx, "used", g(ctx, "spent", ""))
            limit = g(ctx, "limit", "")
            by = g(ctx, "exceeded_by", "")
            cost = _fmt_cost(used)
            pct = ""
            with contextlib.suppress(Exception):
                pct = f"  ({float(used) / float(limit) * 100:.1f}%)"
            return _EventRecord(
                ts,
                hook_name,
                badge="[bold red]⚠ BUDGET EXCEEDED[/bold red]",
                brief=f"{cost} / ${limit}{pct}",
                category=_CAT_BUDGET,
                detail_lines=[
                    "[bold red]⚠ Budget exceeded[/bold red]",
                    f"  used:        {cost}",
                    f"  limit:       ${limit}",
                    f"  exceeded_by: {by or '—'}",
                ],
            )

        if "budget.threshold" in h and self.show_budget:
            # hook emits: threshold_percent, current_value, limit_value
            pct = g(ctx, "threshold_percent", g(ctx, "percent", ""))
            current = _fmt_cost(g(ctx, "current_value", g(ctx, "used", "")))
            limit = g(ctx, "limit_value", g(ctx, "limit", ""))
            return _EventRecord(
                ts,
                hook_name,
                badge=f"[bold orange3]▲ BUDGET {pct}%[/bold orange3]",
                brief=f"{current}" + (f" / ${limit}" if limit else ""),
                category=_CAT_BUDGET,
                detail_lines=[
                    f"[bold orange3]▲ Budget threshold {pct}%[/bold orange3]",
                    f"  spent: {current or '—'}",
                    f"  limit: {'$' + limit if limit else '—'}",
                ],
            )

        if "budget.check" in h and self.show_budget:
            used = g(ctx, "used", g(ctx, "spent", ""))
            remaining = g(ctx, "remaining", "")
            total = g(ctx, "total", "")
            return _EventRecord(
                ts,
                hook_name,
                badge="[dim green]$ budget check[/dim green]",
                brief=f"used=${used}  remaining=${remaining}",
                category=_CAT_BUDGET,
                detail_lines=[
                    "[dim green]$ Budget check[/dim green]",
                    f"  used:      ${used or '—'}",
                    f"  remaining: ${remaining or '—'}",
                    f"  total:     ${total or '—'}",
                ],
            )

        if "ratelimit.exceeded" in h:
            metric = g(ctx, "metric", "")
            limit = g(ctx, "limit", "")
            current = g(ctx, "current", "")
            return _EventRecord(
                ts,
                hook_name,
                badge="[bold red]⚠ RATE LIMIT[/bold red]",
                brief=f"metric={metric}  limit={limit}",
                category=_CAT_BUDGET,
                detail_lines=[
                    "[bold red]⚠ Rate limit exceeded[/bold red]",
                    f"  metric:  {metric or '—'}",
                    f"  limit:   {limit or '—'}",
                    f"  current: {current or '—'}",
                ],
            )

        if "ratelimit.check" in h:
            rpm = g(ctx, "current_rpm", g(ctx, "rpm", ""))
            tpm = g(ctx, "current_tpm", g(ctx, "tpm", ""))
            return _EventRecord(
                ts,
                hook_name,
                badge="[dim green]⏱ rate check[/dim green]",
                brief=f"rpm={rpm}  tpm={tpm}",
                category=_CAT_BUDGET,
                detail_lines=[
                    "[dim green]⏱ Rate limit check[/dim green]",
                    f"  current rpm: {rpm or '—'}",
                    f"  current tpm: {tpm or '—'}",
                ],
            )

        if "ratelimit.threshold" in h:
            pct = g(ctx, "percent", "")
            return _EventRecord(
                ts,
                hook_name,
                badge=f"[orange3]▲ RATE LIMIT {pct}%[/orange3]",
                brief="",
                category=_CAT_BUDGET,
                detail_lines=[f"[orange3]▲ Rate limit threshold {pct}%[/orange3]"],
            )

        # ── Agent Lifecycle ──────────────────────────────────────────────────
        if "agent.run.start" in h:
            inp = g(ctx, "input", "")
            model = g(ctx, "model", "")
            if self.redact_prompts:
                inp = "[redacted]"
            brief_parts = []
            if model:
                brief_parts.append(model)
            brief_parts.append(_trunc(inp, 45))
            return _EventRecord(
                ts,
                hook_name,
                badge="[bold cyan]▶ RUN[/bold cyan]",
                brief="  ".join(p for p in brief_parts if p),
                category=_CAT_AGENT,
                detail_lines=[
                    "[bold cyan]▶ Agent run start[/bold cyan]",
                    f"  model: {model or '—'}",
                ],
                full_content=inp,
            )

        if "agent.run.end" in h:
            iters = g(ctx, "iterations", "?")
            cost = _fmt_cost(g(ctx, "cost", ""))
            dur = _fmt_dur(g(ctx, "duration", ""))
            stop = g(ctx, "stop_reason", "")
            output = g(ctx, "output", g(ctx, "response", g(ctx, "result", "")))
            brief_parts = [f"iter={iters}"]
            if cost:
                brief_parts.append(cost)
            if dur:
                brief_parts.append(dur)
            if stop:
                brief_parts.append(stop)
            return _EventRecord(
                ts,
                hook_name,
                badge="[bold cyan]■ END[/bold cyan]",
                brief="  ".join(brief_parts),
                category=_CAT_AGENT,
                detail_lines=[
                    "[bold cyan]■ Agent run end[/bold cyan]",
                    f"  iterations: {iters}",
                    f"  cost:       {cost or '—'}",
                    f"  duration:   {dur or '—'}",
                    f"  stop:       {stop or '—'}",
                ],
                full_content=output,
            )

        if "handoff.start" in h:
            # hook emits: source_agent, target_agent, user_input, handoff_context
            source = g(ctx, "source_agent", g(ctx, "source", "?"))
            target = g(ctx, "target_agent", g(ctx, "target", "?"))
            task = g(ctx, "user_input", g(ctx, "task", g(ctx, "input", "")))
            snap_obj = self._raw(ctx, "handoff_context")
            snap_lines, snap_full = self._fmt_context_snapshot(snap_obj)
            detail: list[str] = [
                f"[cyan]⇢ Handoff: {source} → {target}[/cyan]",
                f"  from:    [bold]{source}[/bold]",
                f"  to:      [bold]{target}[/bold]",
                f"  input:   {task}",  # full, not truncated
            ]
            detail.extend(snap_lines)
            return _EventRecord(
                ts,
                hook_name,
                badge=f"[cyan]⇢ HANDOFF {source}→{target}[/cyan]",
                brief=_trunc(task, 45),
                category=_CAT_AGENT,
                detail_lines=detail,
                full_content=snap_full or task,
            )

        if "handoff.end" in h:
            # hook emits: cost, response_preview, target_agent
            target = g(ctx, "target_agent", g(ctx, "target", "?"))
            cost = _fmt_cost(g(ctx, "cost", ""))
            preview = g(ctx, "response_preview", g(ctx, "result", ""))
            return _EventRecord(
                ts,
                hook_name,
                badge=f"[cyan]⇠ HANDOFF ← {target}[/cyan]",
                brief=f"{cost}  {_trunc(preview, 40)}",
                category=_CAT_AGENT,
                detail_lines=[
                    f"[cyan]⇠ Handoff done ← {target}[/cyan]",
                    f"  cost:    {cost or '—'}",
                ],
                full_content=preview or "",
            )

        if "handoff.blocked" in h:
            reason = g(ctx, "reason", g(ctx, "message", ""))
            snap_obj = self._raw(ctx, "handoff_context")
            snap_lines, snap_full = self._fmt_context_snapshot(snap_obj)
            detail_b: list[str] = [
                "[bold red]⊘ Handoff blocked[/bold red]",
                f"  reason: {reason}",
            ]
            detail_b.extend(snap_lines)
            return _EventRecord(
                ts,
                hook_name,
                badge="[bold red]⊘ HANDOFF BLOCKED[/bold red]",
                brief=_trunc(reason, 50),
                category=_CAT_AGENT,
                detail_lines=detail_b,
                full_content=snap_full or reason,
            )

        if "spawn.start" in h:
            # hook emits: source_agent, child_agent, input_preview
            source = g(ctx, "source_agent", "")
            name = g(ctx, "child_agent", g(ctx, "agent_name", g(ctx, "name", "?")))
            task = g(ctx, "input_preview", g(ctx, "task", g(ctx, "input", "")))
            return _EventRecord(
                ts,
                hook_name,
                badge=f"[cyan]⊕ SPAWN {name}[/cyan]",
                brief=_trunc(task, 45),
                category=_CAT_AGENT,
                detail_lines=[
                    f"[cyan]⊕ Spawn: {name}[/cyan]",
                    f"  source: {source or '—'}",
                    f"  input:  {_trunc(task, 120)}",
                ],
            )

        if "spawn.end" in h:
            name = g(ctx, "child_agent", g(ctx, "agent_name", g(ctx, "name", "?")))
            cost = _fmt_cost(g(ctx, "cost", ""))
            return _EventRecord(
                ts,
                hook_name,
                badge=f"[cyan]✓ SPAWN {name}[/cyan]",
                brief=cost,
                category=_CAT_AGENT,
                detail_lines=[
                    f"[cyan]✓ Spawn done: {name}[/cyan]",
                    f"  cost: {cost or '—'}",
                ],
            )

        # ── Pipelines ────────────────────────────────────────────────────────
        if "dynamic.pipeline.plan" in h:
            steps = g(ctx, "step_count", g(ctx, "steps", "?"))
            plan = g(ctx, "plan", "")
            return _EventRecord(
                ts,
                hook_name,
                badge="[cyan]⊞ PIPELINE PLAN[/cyan]",
                brief=f"steps={steps}",
                category=_CAT_AGENT,
                detail_lines=[
                    "[cyan]⊞ Dynamic pipeline plan[/cyan]",
                    f"  steps: {steps}",
                    f"  plan:  {_trunc(plan, 100)}",
                ],
            )

        if "dynamic.pipeline.agent.spawn" in h or "pipeline.agent.start" in h:
            name = g(ctx, "agent_name", g(ctx, "name", "?"))
            step = g(ctx, "step", "")
            return _EventRecord(
                ts,
                hook_name,
                badge=f"[dim cyan]  ⊕ pipeline.{name}[/dim cyan]",
                brief=f"step={step}" if step else "",
                category=_CAT_AGENT,
                detail_lines=[
                    f"[dim cyan]  ⊕ Pipeline agent: {name}[/dim cyan]",
                    f"  step: {step or '—'}",
                ],
            )

        if "dynamic.pipeline.agent.complete" in h or "pipeline.agent.complete" in h:
            name = g(ctx, "agent_name", g(ctx, "name", "?"))
            return _EventRecord(
                ts,
                hook_name,
                badge=f"[dim cyan]  ✓ pipeline.{name}[/dim cyan]",
                brief="",
                category=_CAT_AGENT,
                detail_lines=[f"[dim cyan]  ✓ Pipeline agent done: {name}[/dim cyan]"],
            )

        if "dynamic.pipeline.error" in h:
            err = g(ctx, "error", g(ctx, "message", ""))
            return _EventRecord(
                ts,
                hook_name,
                badge="[bold red]✗ PIPELINE ERROR[/bold red]",
                brief=_trunc(err, 50),
                category=_CAT_AGENT,
                detail_lines=[
                    "[bold red]✗ Pipeline error[/bold red]",
                    f"  error: {_trunc(err, 120)}",
                ],
            )

        if "dynamic.pipeline.end" in h or "pipeline.end" in h:
            dur = _fmt_dur(g(ctx, "duration", ""))
            status = g(ctx, "status", "done")
            return _EventRecord(
                ts,
                hook_name,
                badge="[cyan]■ PIPELINE END[/cyan]",
                brief=f"{status}  {dur}",
                category=_CAT_AGENT,
                detail_lines=[
                    "[cyan]■ Pipeline end[/cyan]",
                    f"  status:   {status}",
                    f"  duration: {dur or '—'}",
                ],
            )

        if (
            "pipeline.start" in h
            or "dynamic.pipeline.start" in h
            or "dynamic.pipeline.execute" in h
        ):
            step = g(ctx, "step", g(ctx, "step_name", ""))
            return _EventRecord(
                ts,
                hook_name,
                badge=f"[cyan]▶ PIPELINE {step}[/cyan]",
                brief="",
                category=_CAT_AGENT,
                detail_lines=[f"[cyan]▶ Pipeline: {step or hook_name}[/cyan]"],
            )

        # ── Knowledge & RAG ──────────────────────────────────────────────────
        if "knowledge.search.start" in h:
            query = g(ctx, "query", "")
            if self.redact_prompts:
                query = "[redacted]"
            return _EventRecord(
                ts,
                hook_name,
                badge="[green]🔍 SEARCH[/green]",
                brief=_trunc(query, 60),
                category=_CAT_KNOWLEDGE,
                detail_lines=[
                    "[green]🔍 Knowledge search[/green]",
                    f"  query: {_trunc(query, 120)}",
                ],
            )

        if "knowledge.search.end" in h:
            # hook emits: result_count (singular)
            count = g(
                ctx, "result_count", g(ctx, "results_count", g(ctx, "count", g(ctx, "hits", "?")))
            )
            dur = _fmt_dur(g(ctx, "duration", ""))
            query = g(ctx, "query", "")
            return _EventRecord(
                ts,
                hook_name,
                badge="[bold green]✓ SEARCH[/bold green]",
                brief=f"hits={count}  {dur}",
                category=_CAT_KNOWLEDGE,
                detail_lines=[
                    "[bold green]✓ Knowledge search done[/bold green]",
                    f"  hits:     {count}",
                    f"  duration: {dur or '—'}",
                    f"  query:    {_trunc(query, 80)}",
                ],
            )

        if "knowledge.ingest.start" in h:
            source = g(ctx, "source", g(ctx, "path", ""))
            return _EventRecord(
                ts,
                hook_name,
                badge="[green]📥 INGEST[/green]",
                brief=_trunc(source, 60),
                category=_CAT_KNOWLEDGE,
                detail_lines=[
                    "[green]📥 Knowledge ingest start[/green]",
                    f"  source: {_trunc(source, 120)}",
                ],
            )

        if "knowledge.ingest.end" in h:
            chunks = g(ctx, "chunks", g(ctx, "count", "?"))
            dur = _fmt_dur(g(ctx, "duration", ""))
            return _EventRecord(
                ts,
                hook_name,
                badge="[bold green]✓ INGEST[/bold green]",
                brief=f"chunks={chunks}  {dur}",
                category=_CAT_KNOWLEDGE,
                detail_lines=[
                    "[bold green]✓ Knowledge ingest done[/bold green]",
                    f"  chunks:   {chunks}",
                    f"  duration: {dur or '—'}",
                ],
            )

        if "knowledge.agentic.decompose" in h:
            query = g(ctx, "query", "")
            parts = g(ctx, "parts", g(ctx, "sub_queries", "?"))
            return _EventRecord(
                ts,
                hook_name,
                badge="[green]◈ DECOMPOSE[/green]",
                brief=f"parts={parts}",
                category=_CAT_KNOWLEDGE,
                detail_lines=[
                    "[green]◈ Agentic RAG decompose[/green]",
                    f"  query: {_trunc(query, 80)}",
                    f"  parts: {parts}",
                ],
            )

        if "knowledge.agentic.grade" in h:
            score = g(ctx, "score", g(ctx, "relevance", ""))
            return _EventRecord(
                ts,
                hook_name,
                badge="[green]★ GRADE[/green]",
                brief=f"score={score}",
                category=_CAT_KNOWLEDGE,
                detail_lines=["[green]★ Agentic RAG grade[/green]", f"  score: {score or '—'}"],
            )

        if "knowledge.agentic.refine" in h:
            it = g(ctx, "iteration", "?")
            return _EventRecord(
                ts,
                hook_name,
                badge=f"[green]↺ REFINE #{it}[/green]",
                brief="",
                category=_CAT_KNOWLEDGE,
                detail_lines=[f"[green]↺ Agentic RAG refine #{it}[/green]"],
            )

        if "knowledge.agentic.verify" in h:
            verdict = g(ctx, "verdict", g(ctx, "result", ""))
            return _EventRecord(
                ts,
                hook_name,
                badge="[green]✓ VERIFY[/green]",
                brief=verdict,
                category=_CAT_KNOWLEDGE,
                detail_lines=[
                    "[green]✓ Agentic RAG verify[/green]",
                    f"  verdict: {verdict or '—'}",
                ],
            )

        if "grounding.extract" in h:
            facts = g(ctx, "facts_count", g(ctx, "count", "?"))
            return _EventRecord(
                ts,
                hook_name,
                badge="[green]◇ GROUND extract[/green]",
                brief=f"facts={facts}",
                category=_CAT_KNOWLEDGE,
                detail_lines=["[green]◇ Grounding extract[/green]", f"  facts: {facts}"],
            )

        if "grounding.verify" in h:
            fact = g(ctx, "fact", "")
            verdict = g(ctx, "verdict", "")
            conf = g(ctx, "confidence", "")
            source = g(ctx, "source", "")
            icon = "✓" if "pass" in verdict.lower() or "true" in verdict.lower() else "✗"
            color = "green" if icon == "✓" else "red"
            return _EventRecord(
                ts,
                hook_name,
                badge=f"[{color}]{icon} GROUND verify[/{color}]",
                brief=f"conf={conf}  {_trunc(fact, 40)}",
                category=_CAT_KNOWLEDGE,
                detail_lines=[
                    f"[{color}]{icon} Grounding verify[/{color}]",
                    f"  fact:       {_trunc(fact, 80)}",
                    f"  verdict:    {verdict or '—'}",
                    f"  confidence: {conf or '—'}",
                    f"  source:     {_trunc(source, 80)}",
                ],
            )

        if "grounding.complete" in h:
            verified = g(ctx, "verified_count", g(ctx, "count", "?"))
            return _EventRecord(
                ts,
                hook_name,
                badge="[bold green]✓ GROUNDING done[/bold green]",
                brief=f"verified={verified}",
                category=_CAT_KNOWLEDGE,
                detail_lines=[
                    "[bold green]✓ Grounding complete[/bold green]",
                    f"  verified: {verified}",
                ],
            )

        # ── Guardrails ───────────────────────────────────────────────────────
        if "guardrail.blocked" in h:
            stage = g(ctx, "stage", g(ctx, "rule", g(ctx, "guardrail", "")))
            reason = g(ctx, "reason", g(ctx, "message", ""))
            names = g(ctx, "guardrail_names", "")
            return _EventRecord(
                ts,
                hook_name,
                badge="[bold red]⊘ BLOCKED[/bold red]",
                brief=f"{stage}  {_trunc(reason, 45)}",
                category=_CAT_GUARDRAIL,
                detail_lines=[
                    "[bold red]⊘ Guardrail BLOCKED[/bold red]",
                    f"  stage:  {stage or '—'}",
                    f"  names:  {names or '—'}",
                    f"  reason: {_trunc(reason, 120)}",
                ],
            )

        if "guardrail.input" in h:
            # hook emits: text, stage, guardrail_count (no rule/status fields)
            stage = g(ctx, "stage", "input")
            count = g(ctx, "guardrail_count", "")
            text = g(ctx, "text", "")
            if self.redact_prompts:
                text = "[redacted]"
            return _EventRecord(
                ts,
                hook_name,
                badge="[green]✓ guard.in[/green]",
                brief=f"stage={stage}"
                + (f"  n={count}" if count else "")
                + f"  {_trunc(text, 30)}",
                category=_CAT_GUARDRAIL,
                detail_lines=[
                    "[green]✓ Guardrail input check[/green]",
                    f"  stage:  {stage or '—'}",
                    f"  n:      {count or '—'}",
                    f"  text:   {_trunc(text, 120)}",
                ],
            )

        if "guardrail.output" in h:
            # hook emits: text, stage, guardrail_count (no rule/status fields)
            stage = g(ctx, "stage", "output")
            count = g(ctx, "guardrail_count", "")
            text = g(ctx, "text", "")
            if self.redact_prompts:
                text = "[redacted]"
            return _EventRecord(
                ts,
                hook_name,
                badge="[green]✓ guard.out[/green]",
                brief=f"stage={stage}"
                + (f"  n={count}" if count else "")
                + f"  {_trunc(text, 30)}",
                category=_CAT_GUARDRAIL,
                detail_lines=[
                    "[green]✓ Guardrail output check[/green]",
                    f"  stage:  {stage or '—'}",
                    f"  n:      {count or '—'}",
                    f"  text:   {_trunc(text, 120)}",
                ],
            )

        if "guardrail.error" in h:
            err = g(ctx, "error", g(ctx, "message", ""))
            return _EventRecord(
                ts,
                hook_name,
                badge="[bold red]✗ guardrail error[/bold red]",
                brief=_trunc(err, 50),
                category=_CAT_GUARDRAIL,
                detail_lines=[
                    "[bold red]✗ Guardrail error[/bold red]",
                    f"  error: {_trunc(err, 120)}",
                ],
            )

        if "hitl.pending" in h:
            task = g(ctx, "task", g(ctx, "message", ""))
            return _EventRecord(
                ts,
                hook_name,
                badge="[bold yellow]⏸ HITL PENDING[/bold yellow]",
                brief=_trunc(task, 50),
                category=_CAT_GUARDRAIL,
                detail_lines=[
                    "[bold yellow]⏸ Human-in-the-loop: PENDING[/bold yellow]",
                    f"  task: {_trunc(task, 120)}",
                ],
            )

        if "hitl.approved" in h:
            return _EventRecord(
                ts,
                hook_name,
                badge="[bold green]✓ HITL APPROVED[/bold green]",
                brief="",
                category=_CAT_GUARDRAIL,
                detail_lines=["[bold green]✓ Human-in-the-loop: APPROVED[/bold green]"],
            )

        if "hitl.rejected" in h:
            reason = g(ctx, "reason", "")
            return _EventRecord(
                ts,
                hook_name,
                badge="[bold red]✗ HITL REJECTED[/bold red]",
                brief=_trunc(reason, 50),
                category=_CAT_GUARDRAIL,
                detail_lines=[
                    "[bold red]✗ Human-in-the-loop: REJECTED[/bold red]",
                    f"  reason: {_trunc(reason, 120)}",
                ],
            )

        if "injection.detected" in h:
            source = g(ctx, "source", "")
            pattern = g(ctx, "pattern", g(ctx, "match", ""))
            return _EventRecord(
                ts,
                hook_name,
                badge="[bold red]⚠ INJECTION DETECTED[/bold red]",
                brief=source,
                category=_CAT_GUARDRAIL,
                detail_lines=[
                    "[bold red]⚠ Prompt injection detected[/bold red]",
                    f"  source:  {source or '—'}",
                    f"  pattern: {_trunc(pattern, 80)}",
                ],
            )

        if "injection" in h:
            label = h.split(".")[-1]
            return _EventRecord(
                ts,
                hook_name,
                badge=f"[red]⚠ injection.{label}[/red]",
                brief="",
                category=_CAT_GUARDRAIL,
                detail_lines=[f"[red]⚠ {hook_name}[/red]"],
            )

        # ── Context Management ───────────────────────────────────────────────
        if "context.compress" in h or "context.compact" in h:
            # context.compact uses tokens_before/tokens_after
            before = g(ctx, "tokens_before", g(ctx, "initial_tokens", g(ctx, "before_tokens", "?")))
            after = g(ctx, "tokens_after", g(ctx, "final_tokens", g(ctx, "after_tokens", "?")))
            ratio = g(ctx, "compression_ratio", g(ctx, "ratio", ""))
            return _EventRecord(
                ts,
                hook_name,
                badge="[steel_blue1]◑ COMPRESS[/steel_blue1]",
                brief=f"{before}→{after}" + (f"  ratio={ratio}" if ratio else ""),
                category=_CAT_CONTEXT,
                detail_lines=[
                    "[steel_blue1]◑ Context compress[/steel_blue1]",
                    f"  before: {before} tokens",
                    f"  after:  {after} tokens",
                    f"  ratio:  {ratio or '—'}",
                ],
            )

        if "context.threshold" in h:
            pct = g(ctx, "percent", g(ctx, "usage", ""))
            tokens = g(ctx, "tokens", "")
            return _EventRecord(
                ts,
                hook_name,
                badge=f"[yellow]▲ CONTEXT {pct}%[/yellow]",
                brief=f"{tokens} tokens",
                category=_CAT_CONTEXT,
                detail_lines=[
                    f"[yellow]▲ Context threshold {pct}%[/yellow]",
                    f"  tokens: {tokens or '—'}",
                ],
            )

        if "context.snapshot" in h:
            # Payload: {"snapshot": snap.to_dict(), "utilization_pct": float}
            # snap.to_dict() keys: total_tokens, max_tokens, utilization_pct,
            #   breakdown, messages_count, message_preview, why_included, context_rot_risk
            snap_dict = ctx.get("snapshot", {}) if isinstance(ctx, dict) else {}
            sd = snap_dict if isinstance(snap_dict, dict) else {}
            tokens = sd.get("total_tokens", "")
            max_tokens = sd.get("max_tokens", "")
            utilization = sd.get("utilization_pct", "")
            msg_count = sd.get("messages_count", 0)
            breakdown = sd.get("breakdown", {}) if isinstance(sd.get("breakdown"), dict) else {}
            previews = sd.get("message_preview", [])
            risk = sd.get("context_rot_risk", "")
            compacted = sd.get("compacted", False)

            # Sequential snapshot number
            snap_n = sum(1 for e in self._events if "context.snapshot" in e.hook) + 1

            risk_color = {"low": "green", "medium": "yellow", "high": "red"}.get(str(risk), "white")
            pct_str = f"{utilization:.1f}%" if isinstance(utilization, float) else str(utilization)

            context_detail: list[str] = [
                f"[steel_blue1]📷 Context snapshot #{snap_n}[/steel_blue1]",
                f"  tokens:     [bold]{tokens}[/bold] / {max_tokens}  ({pct_str})",
                f"  messages:   {msg_count}",
                f"  rot risk:   [{risk_color}]{risk or '—'}[/{risk_color}]",
            ]
            if compacted:
                context_detail.append(
                    f"  compacted:  [yellow]yes ({sd.get('compact_method', 'unknown')})[/yellow]"
                )
            if breakdown:
                context_detail.append("")
                context_detail.append("[dim]  ─── token breakdown ──────────────────────[/dim]")
                for part in (
                    "system_tokens",
                    "tools_tokens",
                    "memory_tokens",
                    "messages_tokens",
                    "injected_tokens",
                ):
                    val = breakdown.get(part, 0)
                    if val:
                        label = part.replace("_tokens", "").ljust(10)
                        context_detail.append(f"    {label}  {val} tok")
            why = sd.get("why_included", [])
            if why:
                context_detail.append("")
                context_detail.append("[dim]  ─── why included ────────────────────────[/dim]")
                for reason in why:
                    context_detail.append(f"    • {reason}")
            if previews:
                context_detail.append("")
                context_detail.append(
                    f"[dim]  ─── messages ({len(previews)}) ─────────────────────────[/dim]"
                )
                role_color = {
                    "system": "dim yellow",
                    "user": "cyan",
                    "assistant": "green",
                    "tool": "magenta",
                }
                for i, mp in enumerate(previews):
                    if not isinstance(mp, dict):
                        continue
                    role = mp.get("role", "?")
                    rc = role_color.get(role, "white")
                    tok = mp.get("token_count", "")
                    src = mp.get("source", "")
                    snippet = mp.get("content_snippet", "")
                    tok_str = f"[dim]{tok} tok[/dim]" if tok else ""
                    src_str = f"[dim]{src}[/dim]" if src else ""
                    context_detail.append(
                        f"  [{i + 1}] [{rc}]{role:<12}[/{rc}] {tok_str}  {src_str}"
                    )
                    if snippet:
                        for line in snippet.splitlines()[:4]:
                            context_detail.append(f"      [dim]{line}[/dim]")

            # Build scrollable full content: full messages from raw_messages if present
            # (raw_messages is populated since we store it in _build_snapshot)
            raw_msgs = sd.get("raw_messages") if "raw_messages" in sd else None
            full_parts: list[str] = [
                f"=== context snapshot #{snap_n} ===",
                f"tokens: {tokens} / {max_tokens}  ({pct_str} utilized)",
                f"messages: {msg_count}",
                "",
            ]
            if raw_msgs and isinstance(raw_msgs, list):
                full_parts.append("=== full messages ===")
                full_parts.append("")
                for i, msg in enumerate(raw_msgs):
                    role = msg.get("role", "?") if isinstance(msg, dict) else "?"
                    content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
                    if isinstance(content, list):
                        content = "\n".join(
                            str(p.get("text", p.get("content", str(p))))
                            for p in content
                            if isinstance(p, dict)
                        )
                    full_parts.append(f"[{i + 1}] {role.upper()}")
                    full_parts.append(str(content))
                    full_parts.append("")
            elif previews:
                full_parts.append("=== message previews (snippets only) ===")
                full_parts.append("")
                for i, mp in enumerate(previews):
                    if not isinstance(mp, dict):
                        continue
                    full_parts.append(f"[{i + 1}] {mp.get('role', '?').upper()}")
                    full_parts.append(mp.get("content_snippet", ""))
                    full_parts.append("")

            return _EventRecord(
                ts,
                hook_name,
                badge=f"[steel_blue1]📷 SNAPSHOT #{snap_n}[/steel_blue1]",
                brief=f"{tokens}tok  {msg_count}msgs  {pct_str}",
                category=_CAT_CONTEXT,
                detail_lines=context_detail,
                full_content="\n".join(full_parts),
            )

        if "context.offload" in h:
            tok = g(ctx, "offloaded_tokens", g(ctx, "turns", ""))
            tgt = g(ctx, "offload_target", "")
            return _EventRecord(
                ts,
                hook_name,
                badge="[steel_blue1]⬆ OFFLOAD[/steel_blue1]",
                brief=f"{tok}" + (f"  → {tgt}" if tgt else ""),
                category=_CAT_CONTEXT,
                detail_lines=[
                    "[steel_blue1]⬆ Context offload[/steel_blue1]",
                    f"  tokens: {tok or '—'}",
                    f"  target: {tgt or '—'}",
                ],
            )

        if "context.restore" in h:
            tok = g(ctx, "restored_tokens", g(ctx, "turns", ""))
            source = g(ctx, "source", "")
            return _EventRecord(
                ts,
                hook_name,
                badge="[steel_blue1]⬇ RESTORE[/steel_blue1]",
                brief=f"{tok}" + (f"  from {source}" if source else ""),
                category=_CAT_CONTEXT,
                detail_lines=[
                    "[steel_blue1]⬇ Context restore[/steel_blue1]",
                    f"  tokens: {tok or '—'}",
                    f"  source: {source or '—'}",
                ],
            )

        if "checkpoint.save" in h:
            cid = g(ctx, "checkpoint_id", g(ctx, "path", ""))
            ts_cp = g(ctx, "timestamp", "")
            size = g(ctx, "state_size", "")
            return _EventRecord(
                ts,
                hook_name,
                badge="[steel_blue1]💾 CHECKPOINT SAVE[/steel_blue1]",
                brief=_trunc(cid, 35),
                category=_CAT_CONTEXT,
                detail_lines=[
                    "[steel_blue1]💾 Checkpoint saved[/steel_blue1]",
                    f"  id:        {_trunc(cid, 80)}",
                    f"  timestamp: {ts_cp or '—'}",
                    f"  size:      {size or '—'}",
                ],
            )

        if "checkpoint.load" in h:
            cid = g(ctx, "checkpoint_id", g(ctx, "path", ""))
            ts_cp = g(ctx, "timestamp", "")
            return _EventRecord(
                ts,
                hook_name,
                badge="[steel_blue1]⬇ CHECKPOINT LOAD[/steel_blue1]",
                brief=_trunc(cid, 50),
                category=_CAT_CONTEXT,
                detail_lines=[
                    "[steel_blue1]⬇ Checkpoint loaded[/steel_blue1]",
                    f"  id:        {_trunc(cid, 80)}",
                    f"  timestamp: {ts_cp or '—'}",
                ],
            )

        # ── Output Validation ────────────────────────────────────────────────
        if "output.validation.success" in h:
            attempt = g(ctx, "attempt", "")
            otype = g(ctx, "output_type", g(ctx, "schema", g(ctx, "model", "")))
            fields = g(ctx, "parsed_fields", "")
            return _EventRecord(
                ts,
                hook_name,
                badge="[bold green]✓ OUTPUT valid[/bold green]",
                brief=otype,
                category=_CAT_OUTPUT,
                detail_lines=[
                    "[bold green]✓ Output validation: PASS[/bold green]",
                    f"  type:    {otype or '—'}",
                    f"  attempt: {attempt or '—'}",
                    f"  fields:  {fields or '—'}",
                ],
            )

        if "output.validation.retry" in h:
            attempt = g(ctx, "attempt", "?")
            err = g(ctx, "error", g(ctx, "reason", g(ctx, "message", "")))
            return _EventRecord(
                ts,
                hook_name,
                badge=f"[yellow]↺ OUTPUT retry #{attempt}[/yellow]",
                brief=_trunc(err, 45),
                category=_CAT_OUTPUT,
                detail_lines=[
                    f"[yellow]↺ Output validation retry #{attempt}[/yellow]",
                    f"  error: {_trunc(err, 120)}",
                ],
            )

        if "output.validation.failed" in h:
            err = g(ctx, "error", g(ctx, "reason", g(ctx, "message", "")))
            attempt = g(ctx, "attempt", "?")
            return _EventRecord(
                ts,
                hook_name,
                badge="[bold red]✗ OUTPUT FAILED[/bold red]",
                brief=f"after attempt {attempt}",
                category=_CAT_OUTPUT,
                detail_lines=[
                    "[bold red]✗ Output validation failed[/bold red]",
                    f"  attempt: {attempt}",
                    f"  error:   {_trunc(err, 120)}",
                ],
            )

        if "output.validation.attempt" in h:
            attempt = g(ctx, "attempt", "?")
            return _EventRecord(
                ts,
                hook_name,
                badge=f"[dim]⊡ OUTPUT attempt #{attempt}[/dim]",
                brief="",
                category=_CAT_OUTPUT,
                detail_lines=[f"[dim]⊡ Output validation attempt #{attempt}[/dim]"],
            )

        if "output.validation.start" in h:
            schema = g(ctx, "schema", g(ctx, "model", ""))
            return _EventRecord(
                ts,
                hook_name,
                badge="[blue]⊡ OUTPUT validate[/blue]",
                brief=schema,
                category=_CAT_OUTPUT,
                detail_lines=[
                    "[blue]⊡ Output validation start[/blue]",
                    f"  schema: {schema or '—'}",
                ],
            )

        if "output.validation.error" in h:
            err = g(ctx, "error", "")
            return _EventRecord(
                ts,
                hook_name,
                badge="[bold red]✗ OUTPUT ERROR[/bold red]",
                brief=_trunc(err, 50),
                category=_CAT_OUTPUT,
                detail_lines=[
                    "[bold red]✗ Output validation error[/bold red]",
                    f"  error: {_trunc(err, 120)}",
                ],
            )

        if "system_prompt.after_resolve" in h:
            resolved = g(ctx, "resolved", "")
            length = str(len(resolved)) if resolved else g(ctx, "length", g(ctx, "tokens", ""))
            return _EventRecord(
                ts,
                hook_name,
                badge="[dim]≡ system_prompt resolved[/dim]",
                brief=f"{length}c",
                category=_CAT_OUTPUT,
                detail_lines=[
                    "[dim]≡ System prompt resolved[/dim]",
                    f"  length:  {length or '—'}",
                    f"  preview: {_trunc(resolved, 100)}",
                ],
                full_content=resolved,
            )

        if "system_prompt" in h:
            return _EventRecord(
                ts,
                hook_name,
                badge=f"[dim]≡ {hook_name}[/dim]",
                brief="",
                category=_CAT_OUTPUT,
                detail_lines=[f"[dim]≡ {hook_name}[/dim]"],
            )

        return None

    # ------------------------------------------------------------------
    # Internal — Rich TUI
    # ------------------------------------------------------------------

    def _start_rich(self) -> None:
        try:
            import atexit

            from rich.console import Console
            from rich.live import Live

            real_stdout = getattr(sys, "__stdout__", None) or sys.stdout
            console = Console(file=real_stdout, force_terminal=True)

            self._live = Live(
                _DashboardRenderable(self),
                console=console,
                screen=True,  # alternate screen — handles resize cleanly
                refresh_per_second=8,
                auto_refresh=True,
            )
            self._live.start()

            self._old_stdout = sys.stdout
            sys.stdout = _LiveStdout(self._live, self)

            self._start_key_thread()

            # Keep the panel alive if the script exits without calling detach().
            # atexit handlers run on the main thread while daemon threads (key loop)
            # are still alive, so key input keeps working.
            atexit.register(self._atexit_handler)
        except ImportError:
            self.json_fallback = True

    def _atexit_handler(self) -> None:
        """Block process exit until the user presses q.

        The TUI stays fully live and interactive during this wait — navigation,
        tab switching, and detail views all continue to work.

        Crash-safety: ``stop()`` now stops Rich Live *before* setting
        ``_stop_event``, so by the time our ``_stop_event.wait()`` returns here
        the Rich daemon refresh thread is already dead.  Python's interpreter
        finalisation therefore cannot race with a live daemon writing to stdout,
        which eliminates the::

            Fatal Python error: _enter_buffered_busy: could not acquire lock

        crash that appeared at interpreter shutdown.
        """
        if not self._started or self._stop_event.is_set():
            return
        # Show a completion badge so the user knows they can press q
        if self._live is not None:
            self._events.append(
                _EventRecord(
                    ts=_now(),
                    hook="session.complete",
                    badge="[bold green]✓ done[/bold green]",
                    brief="press q to exit",
                    category=_CAT_OUTPUT,
                    detail_lines=[
                        "[bold green]✓ Session complete[/bold green]",
                        "  Press [bold]q[/bold] to exit",
                    ],
                )
            )
        # Block here — TUI remains interactive.
        # When user presses q → stop() is called → Live stops → _stop_event set.
        # By the time wait() returns, the Rich daemon is guaranteed to be dead.
        self._stop_event.wait()

    # ------------------------------------------------------------------
    # Internal — render helpers
    # ------------------------------------------------------------------

    @property
    def _panel_rows(self) -> int:
        """Dynamic panel height from terminal size. Adapts to resize automatically."""
        if self._live is not None:
            with contextlib.suppress(Exception):
                h = self._live.console.height
                return max(8, int(h) - 10)
        return _PANEL_ROWS_DEFAULT

    def _pad_rows(self, lines: list[str], n: int = 0) -> list[str]:
        """Return exactly n lines, truncating from the top or padding with blanks."""
        if n == 0:
            n = self._panel_rows
        out = list(lines[-n:]) if len(lines) > n else list(lines)
        out += [""] * (n - len(out))
        return out

    def _scrollable_rows(self, all_lines: list[str], view_key: str) -> list[str]:
        """Return panel_rows lines with scroll offset for category detail views."""
        rows = self._panel_rows
        scroll = self._scroll.get(view_key, 0)
        total = len(all_lines)
        if total <= rows:
            return self._pad_rows(all_lines)
        end = total - scroll
        start = max(0, end - rows)
        end = min(total, start + rows)
        result = list(all_lines[start:end])
        if start > 0:
            result[0] = f"[dim]  ↑ {start} more above[/dim]"
        if scroll > 0:
            result[-1] = f"[dim]  ↓ {scroll} more below[/dim]"
        return self._pad_rows(result)

    def _effective_cursor(self, events: list[_EventRecord]) -> int:
        """Resolve the effective cursor index into *events*.

        ``_cursor == -1`` means auto-follow (always last event).
        Returns -1 when the events list is empty.
        """
        if not events:
            return -1
        if self._cursor == -1:
            return len(events) - 1
        return max(0, min(self._cursor, len(events) - 1))

    def _stream_lines_nav(self) -> list[str]:
        """Stream column — scrollbar + vibrant cursor highlight.

        - Auto-follow (cursor==-1): last event gets a navy-blue background that
          tracks new events automatically.
        - Manual (cursor>=0): selected row gets the vivid steel_blue1 background
          when stream panel is focused, or grey27 when right panel is focused.
        """
        events = [r for r in self._events if _event_matches_filter(r.hook, self.filter_mode)]
        if not events:
            return self._pad_rows(["[dim]Waiting for events…[/dim]"])

        total = len(events)
        cursor = self._effective_cursor(events)
        rows = self._panel_rows
        half = rows // 2
        start = max(0, min(cursor - half, total - rows))
        end = min(total, start + rows)

        raw: list[str] = []
        for i, r in enumerate(events[start:end]):
            abs_idx = start + i
            sl = _truncate_markup(r.stream_line, _MAX_STREAM_COLS - 3)
            if abs_idx == cursor:
                plain = _MARKUP_RE.sub("", sl)
                if self._cursor == -1:
                    # Auto-follow: navy bg, arrow + full text
                    raw.append(
                        f"[{_HL_AUTO}][dim cyan]▸[/dim cyan] [white]{plain}[/white][/{_HL_AUTO}]"
                    )
                elif self._focus == "stream":
                    # Manual + stream focused: vivid steel_blue1
                    raw.append(
                        f"[{_HL_FOCUSED}][bold bright_cyan]▸[/bold bright_cyan] [bold bright_white]{plain}[/bold bright_white][/{_HL_FOCUSED}]"
                    )
                else:
                    # Manual + right focused: muted grey
                    raw.append(f"[{_HL_UNFOCUSED}][white]▸ {plain}[/white][/{_HL_UNFOCUSED}]")
            else:
                raw.append(f"  [dim]{sl}[/dim]")

        if start > 0:
            raw[0] = f"[dim]  ↑ {start} more[/dim]"
        if end < total:
            raw[-1] = f"[dim]  ↓ {total - end} more[/dim]"

        return self._pad_rows(raw)

    # ------------------------------------------------------------------
    # Right panel — event detail (tab "event")
    # ------------------------------------------------------------------

    def _hover_preview_lines(self) -> list[str]:
        """Right panel "event" tab: detail of the cursor-selected stream event."""
        events = [r for r in self._events if _event_matches_filter(r.hook, self.filter_mode)]
        cursor = self._effective_cursor(events)
        if cursor < 0:
            # No events yet — show session stats
            return self._pad_rows(
                [
                    "[dim]── stats ──────────────────────────[/dim]",
                    f"  [bold]cost  [/bold] [green]${self._total_cost:.4f}[/green]"
                    + (
                        f" [dim]/ ${self._budget_limit:.2f}[/dim]" if self._budget_limit > 0 else ""
                    ),
                    f"  [bold]tokens[/bold] [cyan]{self._total_tokens:,}[/cyan]",
                    f"  [bold]events[/bold] {len(self._events)}",
                    "",
                    "[dim]Waiting for events…[/dim]",
                ]
            )

        rec = events[cursor]
        lines: list[str] = [
            f"[bold]Hook:[/bold]  {rec.hook}",
            f"[bold]Time:[/bold]  [dim]{rec.ts}[/dim]",
        ]
        if rec.agent_name:
            lines.append(f"[bold]Agent:[/bold] [cyan]{rec.agent_name}[/cyan]")
        if rec.model_name:
            lines.append(f"[bold]Model:[/bold] [blue]{rec.model_name}[/blue]")
        lines.append("")
        lines.extend(rec.detail_lines)
        if rec.full_content:
            lines.append("")
            lines.append("[dim]─── content preview ──────────────[/dim]")
            for chunk in _wrap_plain_lines(rec.full_content.splitlines()[:6], _MAX_DETAIL_COLS - 2):
                lines.append(f"  [dim]{chunk}[/dim]")
        lines.append("")
        lines.append("[dim]↵ full detail  ← stream  → panel[/dim]")
        total = len(lines)
        scroll = self._right_preview_scroll
        rows = self._panel_rows
        end = total - scroll
        start = max(0, end - rows)
        end = min(total, start + rows)
        visible = list(lines[start:end])
        if start > 0 and visible:
            visible[0] = f"[dim]  ↑ {start} more above[/dim]"
        if end < total and visible:
            visible[-1] = f"[dim]  ↓ {total - end} more below[/dim]"
        return self._pad_rows(visible)

    # ------------------------------------------------------------------
    # Right panel — navigatable item lists (tabs: agents/tools/memory/…)
    # ------------------------------------------------------------------

    def _right_panel_records(self) -> list[_EventRecord]:
        """Return the list of records powering the active right-panel tab."""
        rv = self._right_view
        all_ev = list(self._events)

        if rv == "agents":
            return [r for r in all_ev if r.category == _CAT_AGENT]

        if rv == "tools":
            return [r for r in all_ev if r.category == _CAT_TOOL]

        if rv == "memory":
            # Context-aware: show mem/ctx/knowledge events up to the selected
            # stream event so the user sees state "at that point in time".
            stream_ev = [r for r in all_ev if _event_matches_filter(r.hook, self.filter_mode)]
            cursor = self._effective_cursor(stream_ev)
            cutoff = stream_ev[cursor] if cursor >= 0 and stream_ev else None
            mem_cats = {_CAT_MEMORY, _CAT_CONTEXT, _CAT_KNOWLEDGE}
            if cutoff is not None:
                with contextlib.suppress(ValueError):
                    idx = all_ev.index(cutoff)
                    return [r for r in all_ev[: idx + 1] if r.category in mem_cats]
            return [r for r in all_ev if r.category in mem_cats]

        if rv == "guardrails":
            recs = [r for r in all_ev if r.category == _CAT_GUARDRAIL]
            # Append last LLM response as a "last output" item
            for r in reversed(all_ev):
                if r.category == _CAT_LLM and r.full_content:
                    out_rec = _EventRecord(
                        ts=r.ts,
                        hook=r.hook,
                        badge="[bold medium_purple1]◉ last output[/bold medium_purple1]",
                        brief=_trunc(r.full_content, 50),
                        category=_CAT_OUTPUT,
                        detail_lines=[
                            "[bold medium_purple1]◉ Last Agent Output[/bold medium_purple1]",
                            "",
                            *_wrap_plain_lines(r.full_content.splitlines(), _MAX_DETAIL_COLS - 2),
                        ],
                        full_content=r.full_content,
                    )
                    recs = recs + [out_rec]
                    break
            return recs

        if rv == "debug":
            return [
                r
                for r in all_ev
                if "debug" in r.hook
                or "checkpoint" in r.hook.lower()
                or r.hook == "session.complete"
            ]

        if rv == "errors":
            return [r for r in all_ev if _is_error_record(r)]

        return []

    def _effective_right_cursor(self, recs: list[_EventRecord]) -> int:
        """Resolve right-panel cursor, defaulting to the last item."""
        if not recs:
            return -1
        if self._right_cursor == -1:
            return len(recs) - 1
        return max(0, min(self._right_cursor, len(recs) - 1))

    def _right_cursor_move(self, delta: int) -> None:
        """Move right-panel cursor (delta=-1 up, +1 down). Auto-follow logic mirrors stream."""
        recs = self._right_panel_records()
        total = len(recs)
        if not total:
            return
        if self._right_cursor == -1:
            if delta < 0:
                self._right_cursor = max(0, total - 2)
        else:
            new = self._right_cursor + delta
            if new >= total - 1:
                self._right_cursor = -1
            else:
                self._right_cursor = max(0, new)

    def _right_preview_scroll_move(self, delta: int) -> None:
        """Scroll the right-panel event preview when the event tab is focused."""
        if self._right_view != "event":
            return
        if delta < 0:
            self._right_preview_scroll = min(self._right_preview_scroll + 3, 5000)
        elif delta > 0:
            self._right_preview_scroll = max(0, self._right_preview_scroll - 3)

    def _right_panel_item_lines(
        self, recs: list[_EventRecord], header: list[str] | None = None
    ) -> list[str]:
        """Render a navigatable item list for the right panel with scrollbar.

        Each record shows as a single line (badge + brief).  The selected row
        gets a vivid background matching the focused/unfocused state.
        """
        available = self._panel_rows - (len(header) if header else 0)

        if not recs:
            label = self._right_view
            h = list(header) if header else []
            return self._pad_rows(h + [f"[dim]No {label} events yet.[/dim]"])

        total = len(recs)
        cursor = self._effective_right_cursor(recs)
        half = available // 2
        start = max(0, min(cursor - half, total - available))
        end = min(total, start + available)

        raw: list[str] = []
        for i, r in enumerate(recs[start:end]):
            abs_idx = start + i
            sl = _truncate_markup(r.stream_line, _MAX_DETAIL_COLS - 3)
            if abs_idx == cursor:
                plain = _MARKUP_RE.sub("", sl)
                if self._right_cursor == -1:
                    # Auto-follow last: navy bg
                    raw.append(
                        f"[{_HL_AUTO}][dim cyan]▸[/dim cyan] [white]{plain}[/white][/{_HL_AUTO}]"
                    )
                elif self._focus == "right":
                    # Focused + selected: vibrant steel_blue1
                    raw.append(
                        f"[{_HL_FOCUSED}][bold bright_cyan]▸[/bold bright_cyan] [bold bright_white]{plain}[/bold bright_white][/{_HL_FOCUSED}]"
                    )
                else:
                    # Unfocused: muted grey
                    raw.append(f"[{_HL_UNFOCUSED}][white]▸ {plain}[/white][/{_HL_UNFOCUSED}]")
            else:
                raw.append(f"  [dim]{sl}[/dim]")

        if start > 0:
            raw[0] = f"[dim]  ↑ {start} more[/dim]"
        if end < total:
            raw[-1] = f"[dim]  ↓ {total - end} more[/dim]"

        return self._pad_rows(list(header or []) + raw)

    def _debug_tab_lines(self) -> list[str]:
        """Debug tab: current execution state banner + navigatable checkpoint list."""
        recs = self._right_panel_records()  # already filtered to debug events

        # Status banner
        if self._paused and self._step_mode:
            banner = "[bold yellow]▶ STEP MODE[/bold yellow]"
        elif self._paused:
            banner = "[bold red]⏸ PAUSED[/bold red]  [dim]p resume  n step[/dim]"
        else:
            it_recs = [r for r in self._events if "agent.iteration" in r.hook]
            if it_recs:
                last_it = it_recs[-1]
                banner = f"[bold green]● Running[/bold green]  [dim]{last_it.brief}[/dim]"
            else:
                banner = "[bold green]● Running[/bold green]"

        header = [banner, "[dim]── checkpoints ─────────────────────[/dim]"]

        if not recs:
            all_lines = header + [
                "",
                "[dim]No debug points yet.[/dim]",
                "[dim]Add with:  pry.debugpoint('label')[/dim]",
            ]
            return self._pad_rows(all_lines)

        return self._right_panel_item_lines(recs, header=header)

    def _guardrails_output_lines(self) -> list[str]:
        """Guardrails tab: guardrail records (navigatable) + last agent output section."""
        recs = self._right_panel_records()  # includes synthetic "last output" record

        # Split: guardrail records vs last-output record
        guard_recs = [r for r in recs if r.category == _CAT_GUARDRAIL]

        # Count items for the header
        header = [
            f"[bold red]⊘ Guardrails[/bold red]  [dim]{len(guard_recs)} checks[/dim]",
            "[dim]────────────────────────────────────[/dim]",
        ]
        return self._right_panel_item_lines(recs, header=header)

    def _right_panel_lines(self) -> list[str]:
        """Dispatch to the right panel renderer for the active tab."""
        rv = self._right_view
        if rv == "event":
            return self._hover_preview_lines()
        if rv == "debug":
            return self._debug_tab_lines()
        if rv == "guardrails":
            return self._guardrails_output_lines()
        # Generic navigatable item list for all other tabs
        return self._right_panel_item_lines(self._right_panel_records())

    def _status_bar(self) -> str:
        if self._paused and self._step_mode:
            state = "[bold yellow]▶ STEP[/bold yellow]"
        elif self._paused:
            state = "[bold red on grey19] ⏸ PAUSED [/bold red on grey19]"
        else:
            state = "[bold green]●[/bold green]"

        if self._budget_limit > 0:
            pct = min(1.0, self._total_cost / self._budget_limit)
            bc = "red" if pct > 0.8 else "yellow" if pct > 0.5 else "green"
            cost = (
                f"[dim]budget[/dim] "
                f"[{bc}]${self._total_cost:.4f}[/{bc}]"
                f"[dim]/[/dim]"
                f"[dim]${self._budget_limit:.4f}[/dim]"
            )
        else:
            cost = f"[dim]budget[/dim] [green]${self._total_cost:.4f}[/green]"

        model_s = f"  [dim blue]{self._current_model}[/dim blue]" if self._current_model else ""
        if self._input_tokens > 0 or self._output_tokens > 0:
            tok_s = (
                f"  [cyan]in[/cyan][dim]=[/dim][bold cyan]{self._input_tokens:,}[/bold cyan]"
                f" [cyan]out[/cyan][dim]=[/dim][bold cyan]{self._output_tokens:,}[/bold cyan][dim] tok[/dim]"
            )
        elif self._total_tokens > 0:
            tok_s = f"  [dim cyan]{self._total_tokens:,}tok[/dim cyan]"
        else:
            tok_s = ""

        err = f"  [bold red]⚠ {self._error_count} err[/bold red]" if self._error_count else ""
        return f"  {state}  {cost}{tok_s}{model_s}{err}"

    def _hotkey_line(self) -> str:
        """Bottom hotkey bar: 7 tab shortcuts + essential controls."""

        def _k(ch: str, lbl: str, view: str) -> str:
            active = self._right_view == view
            if active:
                color = _TAB_COLOR.get(view, "blue")
                return (
                    f"[bold bright_white on {color}][[bold bright_cyan]{ch}[/bold bright_cyan]][/bold bright_white on {color}]"
                    f"[bold white on {color}] {lbl}[/bold white on {color}]"
                )
            return f"[dim][[/dim][dim cyan]{ch}[/dim cyan][dim]][/dim][dim] {lbl}[/dim]"

        tabs = "  ".join(
            [
                _k("e", "event", "event"),
                _k("a", "agents", "agents"),
                _k("t", "tools", "tools"),
                _k("m", "memory", "memory"),
                _k("g", "guard", "guardrails"),
                _k("d", "debug", "debug"),
                _k("r", "errors", "errors"),
            ]
        )

        pause_lbl = "resume" if self._paused else "pause"
        step_hint = (
            "  [dim][[/dim][bold yellow]n[/bold yellow][dim]] step[/dim]" if self._paused else ""
        )

        controls = (
            "[dim][[/dim][bold cyan]↕[/bold cyan][dim]] nav[/dim]"
            "  [dim][[/dim][bold cyan]↵[/bold cyan][dim]] detail[/dim]"
            "  [dim][[/dim][bold cyan]←→[/bold cyan][dim]] panels[/dim]"
            f"  [dim][[/dim][bold cyan]p[/bold cyan][dim]] {pause_lbl}[/dim]"
            f"{step_hint}"
            "  [dim][[/dim][bold cyan]q[/bold cyan][dim]] quit[/dim]"
        )
        return f"  {tabs}  [dim]│[/dim]  {controls}"

    # ------------------------------------------------------------------
    # Internal — render dispatch
    # ------------------------------------------------------------------

    def _render(self) -> Any:  # type: ignore[explicit-any]
        if self._mode == "detail":
            return self._detail_render()
        if self._mode == "right_detail":
            return self._right_detail_render()
        return self._browse_render()

    def _right_tab_bar(self) -> str:
        """Compact tab bar for the right panel with 7 tabs."""
        _tab_labels: list[tuple[str, str]] = [
            ("e", "event"),
            ("a", "agents"),
            ("t", "tools"),
            ("m", "memory"),
            ("g", "guardrails"),
            ("d", "debug"),
            ("r", "errors"),
        ]
        _SHORT: dict[str, str] = {
            "event": "event",
            "agents": "agents",
            "tools": "tools",
            "memory": "memory",
            "guardrails": "guard",
            "debug": "debug",
            "errors": "errors",
        }
        parts: list[str] = []
        for _key, view in _tab_labels:
            short = _SHORT.get(view, view)
            if self._right_view == view:
                color = _TAB_COLOR.get(view, "blue")
                if self._focus == "right":
                    parts.append(
                        f"[bold bright_white on {color}] {short} [/bold bright_white on {color}]"
                    )
                else:
                    parts.append(f"[bold white on grey23] {short} [/bold white on grey23]")
            else:
                parts.append(f"[dim] {short} [/dim]")
        return " ".join(parts)

    def _browse_render(self) -> Any:  # type: ignore[explicit-any]
        """Two-column browse view.

        Left panel border is bold-cyan when stream is focused, dim otherwise.
        Right panel border uses the tab accent color when right is focused, dim when not.
        """
        try:
            from rich.console import Group as _Group
            from rich.panel import Panel
            from rich.table import Table
            from rich.text import Text

            left_lines = self._stream_lines_nav()
            right_lines = self._right_panel_lines()

            # Focus-dependent border styles
            if self._focus == "stream":
                left_border = "cyan"
                left_title = (
                    "[bold cyan]◉ Stream[/bold cyan]  [dim]↑↓  ↵ detail  → right panel[/dim]"
                )
                right_border = "dim"
            else:
                left_border = "dim"
                left_title = "[dim]◉ Stream[/dim]  [dim]↑↓  ↵ detail  ← focus back[/dim]"
                right_border = _TAB_COLOR.get(self._right_view, "cyan")

            left_panel = Panel(
                Text.from_markup("\n".join(left_lines)),
                title=left_title,
                border_style=left_border,
                padding=(0, 1),
            )
            right_panel = Panel(
                Text.from_markup("\n".join(right_lines)),
                title=self._right_tab_bar(),
                border_style=right_border,
                padding=(0, 1),
            )

            grid = Table.grid(expand=True, padding=0)
            grid.add_column(ratio=58)
            grid.add_column(ratio=42)
            grid.add_row(left_panel, right_panel)

            status = self._status_bar()
            hotkeys = self._hotkey_line()
            divider = "[dim]" + "─" * 72 + "[/dim]"

            extra: list[object] = []
            if self._mode == "input":
                extra.append(
                    Text.from_markup(
                        f"  [bold cyan]?[/bold cyan] {self._input_prompt}"
                        f"[bold white]{self._input_buf}[/bold white][blink]▌[/blink]"
                    )
                )

            content = _Group(
                grid,
                Text.from_markup(status),
                Text.from_markup(divider),
                *extra,  # type: ignore[arg-type]
                Text.from_markup(hotkeys),
            )

            return Panel(
                content,
                title="[bold cyan]pry[/bold cyan]",
                border_style="cyan",
            )
        except Exception:
            from rich.text import Text as _T

            return _T("pry — waiting for events…")

    def _detail_render(self) -> Any:  # type: ignore[explicit-any]
        """Full detail view for the cursor-selected event. Same total height as browse."""
        events = [r for r in self._events if _event_matches_filter(r.hook, self.filter_mode)]
        cursor = self._effective_cursor(events)
        if cursor < 0:
            self._mode = "browse"
            return self._browse_render()

        rec = events[cursor]

        # Build full content lines
        all_lines: list[str] = [
            f"[bold]Hook:[/bold]    {rec.hook}",
            f"[bold]Time:[/bold]    [dim]{rec.ts}[/dim]",
        ]
        if rec.agent_name:
            all_lines.append(f"[bold]Agent:[/bold]   [cyan]{rec.agent_name}[/cyan]")
        if rec.model_name:
            all_lines.append(f"[bold]Model:[/bold]   [bold blue]{rec.model_name}[/bold blue]")
        all_lines.append("")
        all_lines.append("[dim]─── event details ──────────────────────────────────────────[/dim]")
        # Wrap detail lines that would exceed the panel width
        for dl in rec.detail_lines:
            plain = _MARKUP_RE.sub("", dl)
            if len(plain) > _MAX_DETAIL_COLS:
                all_lines.extend(_wrap_plain_lines([plain], _MAX_DETAIL_COLS))
            else:
                all_lines.append(dl)
        if rec.full_content:
            all_lines.append("")
            all_lines.append(
                "[dim]─── full content ───────────────────────────────────────────[/dim]"
            )
            for line in _wrap_plain_lines(rec.full_content.splitlines(), _MAX_DETAIL_COLS - 2):
                all_lines.append(f"  {line}")

        # Scroll within total
        total = len(all_lines)
        scroll = self._detail_scroll
        end = total - scroll
        rows = self._panel_rows
        start = max(0, end - rows)
        end = min(total, start + rows)
        visible = list(all_lines[start:end])
        if start > 0:
            visible[0] = f"[dim]  ↑ {start} more above[/dim]"
        if end < total and visible:
            visible[-1] = f"[dim]  ↓ {total - end} more below[/dim]"
        visible = self._pad_rows(visible)

        # Nav panel (right side)
        total_events = len(events)
        nav_lines = self._pad_rows(
            [
                f"  Event [bold]{cursor + 1}[/bold] of {total_events}",
                "",
                f"  [dim]{rec.ts}[/dim]",
                "",
                "  [dim]─── navigate ─────────────────[/dim]",
                "  [dim white]\\[[/dim white][bold cyan]↑↓[/bold cyan][dim white]][/dim white] [dim]scroll content[/dim]",
                "  [dim white]\\[[/dim white][bold cyan]ESC[/bold cyan][dim white]][/dim white] [dim]back to stream[/dim]",
                "  [dim white]\\[[/dim white][bold cyan]q[/bold cyan][dim white]][/dim white] [dim]quit[/dim]",
            ]
        )

        try:
            from rich.console import Group as _Group
            from rich.panel import Panel
            from rich.table import Table
            from rich.text import Text

            detail_panel = Panel(
                Text.from_markup("\n".join(visible)),
                title=f"[bold cyan]{rec.badge}[/bold cyan]",
                border_style="cyan",
                padding=(0, 1),
            )
            nav_panel = Panel(
                Text.from_markup("\n".join(nav_lines)),
                title="[dim]Navigation[/dim]",
                border_style="dim",
                padding=(0, 1),
            )

            grid = Table.grid(expand=True, padding=0)
            grid.add_column(ratio=65)
            grid.add_column(ratio=35)
            grid.add_row(detail_panel, nav_panel)

            status = self._status_bar()
            hotkeys = (
                "  [dim white]\\[[/dim white][bold cyan]↑↓[/bold cyan][dim white]][/dim white][dim]scroll[/dim]"
                "  [dim white]\\[[/dim white][bold cyan]ESC[/bold cyan][dim white]][/dim white][dim]back[/dim]"
                "  [dim white]\\[[/dim white][bold cyan]q[/bold cyan][dim white]][/dim white][dim]quit[/dim]"
            )
            divider = "[dim]" + "─" * 72 + "[/dim]"

            content = _Group(
                grid,
                Text.from_markup(status),
                Text.from_markup(divider),
                Text.from_markup(hotkeys),
            )

            return Panel(
                content,
                title="[bold cyan]pry  /  detail[/bold cyan]",
                border_style="cyan",
            )
        except Exception:
            from rich.text import Text as _T

            return _T("pry — detail view error")

    def _right_detail_render(self) -> Any:  # type: ignore[explicit-any]
        """Full-screen detail for a right-panel item (Enter on any non-event tab item)."""
        rec = self._right_detail_rec
        if rec is None:
            self._mode = "browse"
            return self._browse_render()

        all_lines: list[str] = [
            f"[bold]Hook:[/bold]    {rec.hook}",
            f"[bold]Time:[/bold]    [dim]{rec.ts}[/dim]",
        ]
        if rec.agent_name:
            all_lines.append(f"[bold]Agent:[/bold]   [cyan]{rec.agent_name}[/cyan]")
        if rec.model_name:
            all_lines.append(f"[bold]Model:[/bold]   [bold blue]{rec.model_name}[/bold blue]")
        all_lines.append("")
        all_lines.append("[dim]─── event details ──────────────────────────────────────────[/dim]")
        for dl in rec.detail_lines:
            plain = _MARKUP_RE.sub("", dl)
            if len(plain) > _MAX_DETAIL_COLS:
                all_lines.extend(_wrap_plain_lines([plain], _MAX_DETAIL_COLS))
            else:
                all_lines.append(dl)
        if rec.full_content:
            all_lines.append("")
            all_lines.append(
                "[dim]─── full content ───────────────────────────────────────────[/dim]"
            )
            for line in _wrap_plain_lines(rec.full_content.splitlines(), _MAX_DETAIL_COLS - 2):
                all_lines.append(f"  {line}")

        total = len(all_lines)
        scroll = self._right_detail_scroll
        end = total - scroll
        rows = self._panel_rows
        start = max(0, end - rows)
        end = min(total, start + rows)
        visible = list(all_lines[start:end])
        if start > 0:
            visible[0] = f"[dim]  ↑ {start} more above[/dim]"
        if end < total and visible:
            visible[-1] = f"[dim]  ↓ {total - end} more below[/dim]"
        visible = self._pad_rows(visible)

        nav_lines = self._pad_rows(
            [
                "",
                f"  [dim]{rec.ts}[/dim]",
                "",
                "  [dim]─── navigate ──────────────────[/dim]",
                "  [dim white][[/dim white][bold cyan]↑↓[/bold cyan][dim white]][/dim white] [dim]scroll[/dim]",
                "  [dim white][[/dim white][bold cyan]ESC[/bold cyan][dim white]][/dim white] [dim]back to panel[/dim]",
                "  [dim white][[/dim white][bold cyan]q[/bold cyan][dim white]][/dim white] [dim]quit[/dim]",
            ]
        )

        try:
            from rich.console import Group as _Group
            from rich.panel import Panel
            from rich.table import Table
            from rich.text import Text

            color = _TAB_COLOR.get(self._right_view, "cyan")
            detail_panel = Panel(
                Text.from_markup("\n".join(visible)),
                title=f"[bold {color}]{rec.badge}[/bold {color}]",
                border_style=color,
                padding=(0, 1),
            )
            nav_panel = Panel(
                Text.from_markup("\n".join(nav_lines)),
                title="[dim]Navigation[/dim]",
                border_style="dim",
                padding=(0, 1),
            )

            grid = Table.grid(expand=True, padding=0)
            grid.add_column(ratio=65)
            grid.add_column(ratio=35)
            grid.add_row(detail_panel, nav_panel)

            hotkeys = (
                "  [dim white][[/dim white][bold cyan]↑↓[/bold cyan][dim white]][/dim white][dim]scroll[/dim]"
                "  [dim white][[/dim white][bold cyan]ESC[/bold cyan][dim white]][/dim white][dim]back[/dim]"
                "  [dim white][[/dim white][bold cyan]q[/bold cyan][dim white]][/dim white][dim]quit[/dim]"
            )
            content = _Group(
                grid,
                Text.from_markup(self._status_bar()),
                Text.from_markup("[dim]" + "─" * 72 + "[/dim]"),
                Text.from_markup(hotkeys),
            )
            return Panel(
                content,
                title=f"[bold cyan]pry  /  {self._right_view}  /  detail[/bold cyan]",
                border_style="cyan",
            )
        except Exception:
            from rich.text import Text as _T

            return _T("pry — right detail view error")

    # ------------------------------------------------------------------
    # Internal — keyboard input
    # ------------------------------------------------------------------

    def _start_key_thread(self) -> None:
        self._stop_event.clear()
        t = threading.Thread(target=self._key_loop, daemon=True)
        t.start()
        self._key_thread = t

    def _key_loop(self) -> None:
        try:
            import os
            import select
            import termios
            import tty

            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            # setcbreak: char-by-char input, keeps OPOST so Rich \n → \r\n works.
            tty.setcbreak(fd)

            def _read1() -> str:
                """Read one raw byte from the terminal fd, bypassing Python buffering."""
                return os.read(fd, 1).decode("utf-8", errors="replace")

            def _ready(timeout: float = 0.05) -> bool:
                r, _, _ = select.select([fd], [], [], timeout)
                return bool(r)

            try:
                while not self._stop_event.is_set():
                    if not _ready(0.1):
                        continue
                    ch = _read1()
                    if ch == "\x1b":
                        if not _ready(0.05):
                            self._handle_key("\x1b")
                            continue

                        ch2 = _read1()
                        if ch2 == "[":
                            if not _ready(0.05):
                                self._handle_key("\x1b")
                                continue
                            ch3 = _read1()
                            self._handle_key(f"\x1b[{ch3}")
                            continue
                        if ch2 == "O":
                            if not _ready(0.05):
                                self._handle_key("\x1b")
                                continue
                            ch3 = _read1()
                            _ss3: dict[str, str] = {
                                "A": "\x1b[A",
                                "B": "\x1b[B",
                                "C": "\x1b[C",
                                "D": "\x1b[D",
                            }
                            self._handle_key(_ss3.get(ch3, f"\x1bO{ch3}"))
                            continue

                        # Unknown / partial escape sequence. Treat it as a bare ESC
                        # so detail views always have a reliable way to go back.
                        self._handle_key("\x1b")
                    else:
                        self._handle_key(ch)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
        except Exception:
            pass  # Non-interactive / Windows — display-only mode

    def _handle_key(self, ch: str) -> None:  # noqa: PLR0912, PLR0915
        # ── Input mode (human-in-the-loop prompt) ────────────────────────────
        if self._mode == "input":
            if ch in ("\r", "\n"):
                self._input_queue.put(self._input_buf)
            elif ch in ("\x7f", "\x08"):
                self._input_buf = self._input_buf[:-1]
            elif ch == "\x1b":
                self._input_queue.put("")
            elif ch.isprintable():
                self._input_buf += ch
            return

        # ── Stream detail mode ───────────────────────────────────────────────
        if self._mode == "detail":
            if ch == "\x1b":
                self._mode = "browse"
                self._detail_scroll = 0
            elif ch == "\x1b[A":
                self._detail_scroll = min(self._detail_scroll + 3, 5000)
            elif ch == "\x1b[B":
                self._detail_scroll = max(0, self._detail_scroll - 3)
            elif ch in ("q", "\x03"):
                self.stop()
            return

        # ── Right-panel item detail mode ─────────────────────────────────────
        if self._mode == "right_detail":
            if ch == "\x1b":
                self._mode = "browse"
                self._right_detail_scroll = 0
                self._right_detail_rec = None
            elif ch == "\x1b[A":
                self._right_detail_scroll = min(self._right_detail_scroll + 3, 5000)
            elif ch == "\x1b[B":
                self._right_detail_scroll = max(0, self._right_detail_scroll - 3)
            elif ch in ("q", "\x03"):
                self.stop()
            return

        # ── Browse mode ───────────────────────────────────────────────────────

        # Letter shortcuts → switch right-panel tab + focus right
        if ch in _RIGHT_KEY_MAP:
            new_view = _RIGHT_KEY_MAP[ch]
            if self._right_view != new_view:
                self._right_cursor = -1  # reset right cursor when switching tab
                self._right_preview_scroll = 0
            self._right_view = new_view
            self._right_view_idx = list(_RIGHT_VIEWS).index(new_view)
            self._focus = "right"

        elif ch == "\t":  # Tab → next right tab
            self._right_view_idx = (self._right_view_idx + 1) % len(_RIGHT_VIEWS)
            self._right_view = _RIGHT_VIEWS[self._right_view_idx]
            self._right_cursor = -1
            self._right_preview_scroll = 0
            self._focus = "right"

        elif ch == "\x1b[Z":  # Shift+Tab → prev right tab
            self._right_view_idx = (self._right_view_idx - 1) % len(_RIGHT_VIEWS)
            self._right_view = _RIGHT_VIEWS[self._right_view_idx]
            self._right_cursor = -1
            self._right_preview_scroll = 0
            self._focus = "right"

        elif ch == "\x1b[C":  # → Right arrow: focus right panel
            self._focus = "right"

        elif ch == "\x1b[D":  # ← Left arrow: focus stream panel
            self._focus = "stream"

        elif ch == "p":
            self.resume() if self._paused else self.pause()

        elif ch == "n":
            self.step()

        elif ch in ("\r", "\n"):  # Enter
            if self._focus == "right":
                if self._right_view != "event":
                    # Enter on a right-panel item → full detail view
                    recs = self._right_panel_records()
                    cursor = self._effective_right_cursor(recs)
                    if cursor >= 0:
                        self._right_detail_rec = recs[cursor]
                        self._right_detail_scroll = 0
                        self._mode = "right_detail"
                # For "event" tab Enter just refocuses stream
                else:
                    self._focus = "stream"
            else:
                # Stream panel: lock cursor then enter detail
                if self._cursor == -1:
                    evts = [
                        r for r in self._events if _event_matches_filter(r.hook, self.filter_mode)
                    ]
                    if evts:
                        self._cursor = len(evts) - 1
                if self._cursor >= 0:
                    self._mode = "detail"
                    self._detail_scroll = 0

        elif ch == "\x1b":  # bare ESC
            if self._focus == "right":
                self._focus = "stream"
            else:
                self._cursor = -1  # back to auto-follow

        elif ch == "\x1b[A":  # ↑ Up arrow
            if self._focus == "right":
                if self._right_view == "event":
                    self._right_preview_scroll_move(-1)
                else:
                    self._right_cursor_move(-1)
            else:
                self._cursor_move(-1)

        elif ch == "\x1b[B":  # ↓ Down arrow
            if self._focus == "right":
                if self._right_view == "event":
                    self._right_preview_scroll_move(1)
                else:
                    self._right_cursor_move(1)
            else:
                self._cursor_move(1)

        elif ch in ("q", "\x03"):
            self.stop()

    def _cursor_move(self, delta: int) -> None:
        """Move the stream cursor (delta<0 = up/older, delta>0 = down/newer)."""
        events = [r for r in self._events if _event_matches_filter(r.hook, self.filter_mode)]
        total = len(events)
        if not total:
            return
        if self._cursor == -1:
            if delta < 0:
                self._cursor = max(0, total - 2)
        else:
            new = self._cursor + delta
            if new >= total - 1:
                self._cursor = -1  # past last → auto-follow
            else:
                self._cursor = max(0, new)
