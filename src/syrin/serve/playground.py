"""Web playground for testing agents — chat UI, streaming, budget gauge, observability.

Provides context-based event collection for observability. Serves the Next.js playground
(playground/out/) when available; falls back to inline HTML otherwise.
"""

from __future__ import annotations

import contextvars
import json
from collections.abc import Generator
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from syrin.agent import Agent


def add_playground_static_mount(app: Any, mount_path: str) -> None:
    """Mount playground static files on the app at the given path.

    Must be called on the main FastAPI app (not an APIRouter) because include_router
    does not transfer Mount objects. Call after include_router(agent.as_router()).
    """
    static_dir = _playground_static_dir()
    if static_dir is None:
        return
    from fastapi.staticfiles import StaticFiles

    path = mount_path.rstrip("/") or "/playground"
    if not path.startswith("/"):
        path = "/" + path
    app.mount(path, StaticFiles(directory=str(static_dir), html=True), name="playground_static")


def _playground_static_dir() -> Path | None:
    """Return path to playground static files (Next.js build) or None if not found.

    Checks: (1) repo playground/out/ (dev — use after npm run build), (2) package
    resource syrin.serve.playground_static (published wheel).
    Prefer playground/out so the server serves the latest build during development.
    """
    here = Path(__file__).resolve()
    repo_root = here.parents[3]  # serve -> syrin -> src -> repo
    out = repo_root / "playground" / "out"
    if out.is_dir():
        return out
    try:
        from importlib.resources import files

        pkg = files("syrin.serve")
        static = pkg / "playground_static"
        if static.is_dir():
            return Path(str(static))
    except (ImportError, TypeError):
        pass
    return None


_playground_events: contextvars.ContextVar[list[tuple[str, dict[str, Any]]] | None] = (
    contextvars.ContextVar("syrin_playground_events", default=None)
)


_TRUNCATE_DATA_URL_AT = 100
_TRUNCATE_STRING_AT = 200


def _truncate_data_urls(obj: Any) -> Any:
    """Truncate long data URLs, content_bytes reprs, and other huge strings in events."""
    if obj is None or isinstance(obj, (bool, int, float)):
        return obj
    if isinstance(obj, bytes):
        return f"<bytes len={len(obj)}>"
    if isinstance(obj, str):
        if len(obj) <= _TRUNCATE_DATA_URL_AT:
            return obj
        if obj.startswith("data:image") or obj.startswith("data:video"):
            return obj[:_TRUNCATE_DATA_URL_AT] + "… [truncated]"
        if len(obj) > _TRUNCATE_STRING_AT:
            return obj[:_TRUNCATE_DATA_URL_AT] + "… [truncated]"
        return obj
    if isinstance(obj, dict):
        return {k: _truncate_data_urls(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_truncate_data_urls(x) for x in obj]
    return obj


def _to_json_safe(obj: Any) -> Any:
    """Convert object to JSON-serializable form (handles datetime, objects)."""
    if obj is None or isinstance(obj, (bool, int, float)):
        return obj
    if isinstance(obj, bytes):
        return _truncate_data_urls(obj)
    if isinstance(obj, str):
        return _truncate_data_urls(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return _truncate_data_urls({k: _to_json_safe(v) for k, v in obj.items()})
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(x) for x in obj]
    # Custom objects (TokenUsage, etc.): convert to dict if has __dict__, else str
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        return _to_json_safe(vars(obj))
    return str(obj)


def _attach_event_collector(agent: Agent) -> None:
    """Register agent.events.on_all() handler that appends (hook.value, ctx dict) to context var.

    Only appends when the context var is set (non-empty list in context). Used with
    _collect_events() so events from the current request are captured.
    """

    def handler(hook: Any, ctx: Any) -> None:
        events = _playground_events.get()
        if events is not None and isinstance(events, list):
            hook_value = getattr(hook, "value", str(hook))
            ctx_dict: dict[str, Any] = dict(ctx) if hasattr(ctx, "items") else {}
            safe = _to_json_safe(ctx_dict)
            events.append((hook_value, safe))

    agent.events.on_all(handler)


@contextmanager
def _collect_events() -> Generator[list[tuple[str, dict[str, Any]]], None, None]:
    """Context manager that sets context var to empty list, yields, then returns the list."""
    events: list[tuple[str, dict[str, Any]]] = []
    token = _playground_events.set(events)
    try:
        yield events
    finally:
        _playground_events.reset(token)


def get_playground_html(
    base_path: str,
    api_base: str,
    agents: list[dict[str, Any]],
    *,
    debug: bool = False,
) -> str:
    """Return full HTML for the playground UI with embedded CSS and JS.

    Args:
        base_path: Base path for the playground page (e.g. /playground).
        api_base: API base for requests. Single agent: "" or route_prefix.
                  Multi-agent: /agent (prefix for /agent/{name}/stream, etc.).
        agents: List of {"name": str, "description": str} dicts. Single agent: one item.
        debug: If True, show observability panel (collapsible) for events from responses.

    Returns:
        Complete HTML document as string.
    """
    api_base = (api_base or "").rstrip("/")
    base = f"{api_base}/" if api_base else "/"
    multi_agent = len(agents) > 1
    agent_options = "".join(
        f'<option value="{a["name"]}">{a.get("description", a["name"])}</option>' for a in agents
    )
    agent_selector = ""
    if multi_agent:
        agent_selector = f"""
        <div class="agent-selector">
          <label for="agent-select">Agent</label>
          <select id="agent-select">{agent_options}</select>
        </div>"""

    debug_panel = ""
    if debug:
        debug_panel = """
        <details class="observability-panel" id="observability-panel">
          <summary>Observability (debug)</summary>
          <div id="events-display" class="events-display"></div>
        </details>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Syrin Playground</title>
  <style>
    :root {{
      --bg: #0f0f12;
      --surface: #1a1a1f;
      --border: #2a2a32;
      --text: #e4e4e7;
      --text-muted: #71717a;
      --accent: #6366f1;
      --accent-hover: #818cf8;
      --user-bubble: #3b82f6;
      --assistant-bubble: #27272a;
      --success: #22c55e;
      --error: #ef4444;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }}
    .container {{
      max-width: 720px;
      margin: 0 auto;
      padding: 1rem;
      flex: 1;
      display: flex;
      flex-direction: column;
      width: 100%;
    }}
    .header {{
      display: flex;
      align-items: center;
      gap: 1rem;
      margin-bottom: 1rem;
    }}
    .agent-selector {{
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }}
    .agent-selector label {{
      color: var(--text-muted);
      font-size: 0.875rem;
    }}
    .agent-selector select {{
      background: var(--surface);
      border: 1px solid var(--border);
      color: var(--text);
      padding: 0.4rem 0.75rem;
      border-radius: 6px;
      font-size: 0.9rem;
    }}
    .chat-area {{
      flex: 1;
      display: flex;
      flex-direction: column;
      min-height: 0;
    }}
    .messages {{
      flex: 1;
      overflow-y: auto;
      padding: 1rem 0;
      display: flex;
      flex-direction: column;
      gap: 1rem;
    }}
    .message {{
      max-width: 90%;
      padding: 0.75rem 1rem;
      border-radius: 12px;
      line-height: 1.5;
      white-space: pre-wrap;
    }}
    .message.user {{
      align-self: flex-end;
      background: var(--user-bubble);
      color: white;
    }}
    .message.assistant {{
      align-self: flex-start;
      background: var(--assistant-bubble);
      border: 1px solid var(--border);
    }}
    .message .meta {{
      margin-top: 0.5rem;
      font-size: 0.75rem;
      color: var(--text-muted);
    }}
    .input-row {{
      display: flex;
      gap: 0.5rem;
      padding: 1rem 0;
      border-top: 1px solid var(--border);
    }}
    .input-row input {{
      flex: 1;
      background: var(--surface);
      border: 1px solid var(--border);
      color: var(--text);
      padding: 0.6rem 1rem;
      border-radius: 8px;
      font-size: 1rem;
    }}
    .input-row input:focus {{
      outline: none;
      border-color: var(--accent);
    }}
    .input-row button {{
      background: var(--accent);
      color: white;
      border: none;
      padding: 0.6rem 1.25rem;
      border-radius: 8px;
      font-size: 1rem;
      cursor: pointer;
    }}
    .input-row button:hover {{
      background: var(--accent-hover);
    }}
    .input-row button:disabled {{
      opacity: 0.5;
      cursor: not-allowed;
    }}
    .budget-gauge {{
      margin: 0.5rem 0;
      padding: 0.5rem;
      background: var(--surface);
      border-radius: 6px;
      font-size: 0.8rem;
      color: var(--text-muted);
    }}
    .budget-gauge.hidden {{ display: none; }}
    .footer {{
      text-align: center;
      padding: 1rem;
      font-size: 0.8rem;
      color: var(--text-muted);
    }}
    .footer a {{
      color: var(--text-muted);
      text-decoration: none;
    }}
    .footer a:hover {{
      color: var(--accent);
    }}
    .observability-panel {{
      margin-top: 1rem;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 0.5rem 1rem;
    }}
    .observability-panel summary {{
      cursor: pointer;
      color: var(--text-muted);
      font-size: 0.875rem;
    }}
    .observability-panel .events-display,
    .observability-panel pre {{
      margin: 0.5rem 0 0;
      font-size: 0.75rem;
      overflow-x: auto;
      white-space: pre-wrap;
      max-height: 280px;
      overflow-y: auto;
    }}
    .event-block {{ margin-bottom: 1rem; border-bottom: 1px solid var(--border); padding-bottom: 0.5rem; }}
    .event-hook {{ color: var(--accent); font-weight: 600; margin-bottom: 0.25rem; }}
    .event-results {{ margin: 0.25rem 0; }}
    .result-item {{ margin: 0.35rem 0; font-size: 0.7rem; }}
    .result-meta {{ color: var(--text-muted); margin-right: 0.25rem; }}
    .expand-btn {{ font-size: 0.65rem; padding: 0.1rem 0.35rem; cursor: pointer; background: var(--border); border: none; color: var(--text); border-radius: 4px; }}
    .expand-btn:hover {{ background: var(--accent); }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">{agent_selector}</div>
    <div class="budget-gauge hidden" id="budget-gauge"></div>
    <div class="chat-area">
      <div class="messages" id="messages"></div>
      <div class="input-row">
        <input type="text" id="input" placeholder="Type a message..." autocomplete="off" />
        <button id="send">Send</button>
      </div>
    </div>
    {debug_panel}
  </div>
  <footer class="footer">
    <a href="https://syrin.ai" target="_blank" rel="noopener">Powered by Syrin</a>
  </footer>
  <script>
(function() {{
  const multiAgent = {str(multi_agent).lower()};
  const debug = {str(debug).lower()};
  const basePath = {json.dumps(base_path)};
  const apiBase = {json.dumps(base)};

  function streamUrl() {{
    if (multiAgent) {{
      const name = document.getElementById("agent-select").value;
      return apiBase + name + "/stream";
    }}
    return apiBase + "stream";
  }}
  function chatUrl() {{
    if (multiAgent) {{
      const name = document.getElementById("agent-select").value;
      return apiBase + name + "/chat";
    }}
    return apiBase + "chat";
  }}
  function budgetUrl() {{
    if (multiAgent) {{
      const name = document.getElementById("agent-select").value;
      return apiBase + name + "/budget";
    }}
    return apiBase + "budget";
  }}

  const messagesEl = document.getElementById("messages");
  const inputEl = document.getElementById("input");
  const sendBtn = document.getElementById("send");
  const budgetEl = document.getElementById("budget-gauge");

  function addMessage(role, content, meta) {{
    const div = document.createElement("div");
    div.className = "message " + role;
    let html = content.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/\\n/g, "<br>");
    div.innerHTML = html;
    if (meta) {{
      const metaEl = document.createElement("div");
      metaEl.className = "meta";
      metaEl.textContent = meta;
      div.appendChild(metaEl);
    }}
    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }}

  function fetchBudget() {{
    fetch(budgetUrl()).then(r => {{
      if (!r.ok) return;
      return r.json();
    }}).then(data => {{
      if (!data) return;
      budgetEl.classList.remove("hidden");
      const pct = data.percent_used != null ? data.percent_used : 0;
      const spent = data.spent != null ? "$" + Number(data.spent).toFixed(4) : "";
      const remaining = data.remaining != null ? "$" + Number(data.remaining).toFixed(4) : "";
      budgetEl.textContent = `Budget: ${{spent}} spent, ${{remaining}} remaining (${{pct.toFixed(1)}}% used)`;
    }}).catch(() => {{}});
  }}

  const TRUNCATE_LEN = 100;
  function escapeHtml(s) {{
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }}
  function renderEvent(ev) {{
    const hook = ev.hook || "";
    const ctx = ev.ctx || {{}};
    let html = '<div class="event-block">';
    html += '<div class="event-hook">' + escapeHtml(hook) + '</div>';
    if (ctx.results && Array.isArray(ctx.results)) {{
      html += '<div class="event-results">';
      ctx.results.forEach((r, i) => {{
        const content = r.content || "";
        const short = content.length <= TRUNCATE_LEN ? content : content.slice(0, TRUNCATE_LEN) + "...";
        html += '<div class="result-item">';
        html += '<span class="result-meta">[' + (r.rank || i+1) + '] score=' + (r.score ?? "") + '</span> ';
        if (content.length > TRUNCATE_LEN) {{
          const rid = "res_" + i + "_" + Math.random().toString(36).slice(2, 8);
          html += '<span id="' + rid + '_s">' + escapeHtml(short) + '</span>';
          html += ' <button type="button" class="expand-btn" data-rid="' + rid + '">+ more</button>';
          html += '<span id="' + rid + '_f" hidden>' + escapeHtml(content) + '</span>';
        }} else {{
          html += escapeHtml(content);
        }}
        html += '</div>';
      }});
      html += '</div>';
    }}
    const rest = {{ ...ctx }};
    delete rest.results;
    if (Object.keys(rest).length) {{
      html += '<pre class="event-ctx">' + escapeHtml(JSON.stringify(rest, null, 2)) + '</pre>';
    }}
    html += '</div>';
    return html;
  }}
  function showEvents(events) {{
    if (!debug) return;
    const panel = document.getElementById("observability-panel");
    const container = document.getElementById("events-display");
    if (!panel || !container) return;
    container.innerHTML = events.map(renderEvent).join("");
    container.classList.add("events-container");
    container.querySelectorAll(".expand-btn").forEach(btn => {{
      btn.addEventListener("click", function() {{
        const rid = this.getAttribute("data-rid");
        const s = document.getElementById(rid + "_s");
        const f = document.getElementById(rid + "_f");
        if (!s || !f) return;
        if (f.hidden) {{ f.hidden = false; s.hidden = true; this.textContent = "- less"; }}
        else {{ f.hidden = true; s.hidden = false; this.textContent = "+ more"; }}
      }});
    }});
    panel.open = true;
  }}

  async function sendMessage() {{
    const text = inputEl.value.trim();
    if (!text) return;
    inputEl.value = "";
    sendBtn.disabled = true;
    addMessage("user", text);

    const assistantDiv = document.createElement("div");
    assistantDiv.className = "message assistant";
    messagesEl.appendChild(assistantDiv);
    messagesEl.scrollTop = messagesEl.scrollHeight;

    let meta = null;
    let lastMeta = null;
    let events = null;

    try {{
      const res = await fetch(streamUrl(), {{
        method: "POST",
        headers: {{ "Content-Type": "application/json" }},
        body: JSON.stringify({{ message: text }})
      }});

      if (!res.ok) {{
        const err = await res.json().catch(() => ({{}}));
        throw new Error(err.error || res.statusText || "Request failed");
      }}

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";

      while (true) {{
        const {{ value, done }} = await reader.read();
        if (done) break;
        buf += decoder.decode(value, {{ stream: true }});
        const lines = buf.split("\\n");
        buf = lines.pop() || "";
        for (const line of lines) {{
          if (line.startsWith("data: ")) {{
            try {{
              const data = JSON.parse(line.slice(6));
              if (data.done) {{
                if (data.cost != null || data.tokens) {{
                  const parts = [];
                  if (data.cost != null) parts.push("$" + Number(data.cost).toFixed(6));
                  if (data.tokens) {{
                    const t = data.tokens;
                    parts.push((t.total || t.output_tokens || 0) + " tokens");
                  }}
                  meta = parts.join(" · ");
                }}
                if (data.events && data.events.length) events = data.events;
              }} else if (data.text != null) {{
                const content = (data.accumulated != null ? data.accumulated : (assistantDiv.textContent || "") + data.text);
                assistantDiv.textContent = content;
                messagesEl.scrollTop = messagesEl.scrollHeight;
                if (data.cost != null || data.tokens) {{
                  const parts = [];
                  if (data.cost != null) parts.push("$" + Number(data.cost).toFixed(6));
                  if (data.tokens) {{
                    const t = data.tokens;
                    const tot = t.total_tokens ?? t.total ?? (t.input_tokens || 0) + (t.output_tokens || 0);
                    if (tot) parts.push(tot + " tokens");
                  }}
                  lastMeta = parts.join(" · ");
                }}
              }}
            }} catch (e) {{}}
          }}
        }}
      }}

      meta = meta || lastMeta;
      if (meta) {{
        const metaEl = document.createElement("div");
        metaEl.className = "meta";
        metaEl.textContent = meta;
        assistantDiv.appendChild(metaEl);
      }}
      if (events) showEvents(events);
      fetchBudget();
    }} catch (err) {{
      assistantDiv.textContent = "Error: " + (err.message || "Unknown error");
      assistantDiv.style.color = "var(--error)";
    }} finally {{
      sendBtn.disabled = false;
    }}
  }}

  sendBtn.addEventListener("click", sendMessage);
  inputEl.addEventListener("keydown", e => {{
    if (e.key === "Enter" && !e.shiftKey) {{ e.preventDefault(); sendMessage(); }}
  }});

  if (multiAgent) {{
    document.getElementById("agent-select").addEventListener("change", fetchBudget);
  }}
  fetchBudget();
}})();
  </script>
</body>
</html>
"""
