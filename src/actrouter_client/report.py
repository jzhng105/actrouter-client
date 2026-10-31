"""What routing saved: a summary of the routing ledger, as text, JSON or HTML.

Two numbers, kept apart because they answer different questions:

* **measured** -- for routed subagents whose outcome was recorded
  (Claude Code ``PostToolUse``), the real token usage is priced twice: on the
  model that actually ran, and on the baseline model that would have run
  without routing. Same tokens, two price lists: this isolates the price
  effect of the switch. (A different model would not use exactly the same
  tokens; the report says so.)
* **projected** -- routed calls with no recorded outcome, priced at the
  median measured usage of the same task type (once at least
  ``PROJECTION_MIN_MEASURED`` of its calls were measured), on both models.
  Real token counts from this ledger, just not this call's own.
* **estimated** -- the router's own before-the-fact estimate for every
  routed call, from the task's token estimate (the caller's, or the task
  type's typical agent work). Covers calls with no outcome (``launch``, MCP
  advice), but is only as good as the estimate. Estimates recorded before
  token priors existed price about 200 tokens a call; the report says how
  many it holds.

Prices come from the catalog (``GET /v1/models``), so cached
prompt tokens and long-context tiers are priced as the router prices them.
Cache *writes* are priced at the input rate; providers charge somewhat more,
so measured costs of cache-heavy calls are slightly low on both sides.
"""

from __future__ import annotations

import datetime as dt
import html
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

from actrouter_client import ledger
from actrouter_client.baseline import to_catalog_id

Usage = dict[str, int]

#: measured calls of a task type needed before its median projects others
PROJECTION_MIN_MEASURED = 2

USAGE_KEYS = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")


# --- pricing ---------------------------------------------------------------------


def _rates(model: dict[str, Any], prompt_tokens: int) -> tuple[Optional[float], Optional[float], Optional[float]]:
    rates = (
        model.get("input_price_per_mtok"),
        model.get("output_price_per_mtok"),
        model.get("cache_read_price_per_mtok"),
    )
    inclusive = bool(model.get("threshold_inclusive"))
    for tier in sorted(model.get("context_tiers") or [], key=lambda t: t["above_tokens"]):
        if prompt_tokens > tier["above_tokens"] or (inclusive and prompt_tokens == tier["above_tokens"]):
            rates = (
                tier.get("input_price_per_mtok"),
                tier.get("output_price_per_mtok"),
                tier.get("cache_read_price_per_mtok"),
            )
    return rates


def cost_of(usage: Usage, model: Optional[dict[str, Any]], prompt_tokens: Optional[int] = None) -> Optional[float]:
    """USD for `usage` on `model` (a catalog listing entry), or None if unpriced.

    `prompt_tokens` is the largest prompt one request sent, which is what a
    long-context tier applies to; without it, `usage` is taken as one request.
    """
    if not model:
        return None
    fresh = usage.get("input_tokens", 0) + usage.get("cache_creation_input_tokens", 0)
    cached = usage.get("cache_read_input_tokens", 0)
    out = usage.get("output_tokens", 0)
    in_rate, out_rate, cache_rate = _rates(model, fresh + cached if prompt_tokens is None else prompt_tokens)
    if in_rate is None or out_rate is None:
        return None
    if cache_rate is None:
        cache_rate = in_rate
    return (fresh * in_rate + cached * cache_rate + out * out_rate) / 1_000_000


# --- summary ---------------------------------------------------------------------


def _usage(outcome: dict[str, Any]) -> Optional[Usage]:
    if not any(isinstance(outcome.get(k), int) for k in USAGE_KEYS):
        return None
    return {k: int(outcome.get(k) or 0) for k in USAGE_KEYS}


def _median_call(calls: list[tuple[Usage, Optional[int]]]) -> tuple[Usage, Optional[int]]:
    """The median-sized call (by total tokens) with its largest request, so
    its token mix is one a real call had."""
    ordered = sorted(calls, key=lambda c: sum(c[0].values()))
    return ordered[(len(ordered) - 1) // 2]


#: why a routed call has no measured usage, and what that means for the reader
UNMEASURED_CAUSES = {
    "unclaimed_advice": "were route_task advice that no subagent claimed: none was started on the advised "
                        "model within 15 minutes, or the agent did the work itself",
    "code_advice": "were model advice for code (route_function): nothing ran to measure",
    "cli_advice": "were `actrouter-route` advice from the command line: nothing ran to measure",
    "launch": "started an agent CLI with `actrouter-route launch`, which reports no usage",
    "background_pending": "are background subagents with no usage yet: still running, or the plugin "
                          "predates its SubagentStop hook (update it)",
    "no_outcome": "are subagents with no PostToolUse outcome: still running, interrupted, or the hook is "
                  "not installed",
    "no_usage": "finished without reporting usage",
}


def _unmeasured_cause(route: dict[str, Any], outcome: Optional[dict[str, Any]]) -> str:
    if outcome is not None:
        return "background_pending" if outcome.get("status") == "async_launched" else "no_usage"
    if route.get("tool_use_id"):
        return "no_outcome"
    surface = route.get("surface")
    if surface == "mcp":
        return "unclaimed_advice" if route.get("tool") in (None, "route_task") else "code_advice"
    return {"launch": "launch", "cli": "cli_advice"}.get(str(surface), "no_outcome")


def _pct(saved: float, baseline: float) -> Optional[float]:
    return round(100 * saved / baseline, 1) if baseline > 0 else None


def summarize(
    events: Iterable[dict[str, Any]],
    catalog: list[dict[str, Any]],
    *,
    source: str = "",
) -> dict[str, Any]:
    """The report's numbers, from ledger events and the model catalog."""
    by_id = {m["id"]: m for m in catalog if m.get("id")}
    routes: list[dict[str, Any]] = []
    outcomes: dict[tuple[Any, Any], dict[str, Any]] = {}
    #: SubagentStop usage, by agent: what a background launch's outcome lacks
    finished: dict[tuple[Any, Any], dict[str, Any]] = {}
    skipped = prompts = 0
    chat_turns: list[dict[str, Any]] = []
    first: Optional[str] = None
    last: Optional[str] = None
    claimed: set[Any] = set()
    for e in events:
        ts = str(e.get("ts") or "")
        first, last = (ts if first is None or ts < first else first), (ts if last is None or ts > last else last)
        if e["kind"] == "route":
            routes.append(e)
            if e.get("claims"):
                claimed.add(e["claims"])
        elif e["kind"] == "skipped":
            skipped += 1
        elif e["kind"] == "prompt":
            prompts += 1
        elif e["kind"] == "turn":
            chat_turns.append(e)
        elif e["kind"] == "outcome" and e.get("tool_use_id"):
            outcomes[(e.get("session_id"), e["tool_use_id"])] = e
        elif e["kind"] == "outcome" and e.get("agent_id"):
            finished[(e.get("session_id"), e["agent_id"])] = e

    measured = {"calls": 0, "actual_usd": 0.0, "baseline_usd": 0.0, "tokens": 0}
    projected = {"calls": 0, "routed_usd": 0.0, "baseline_usd": 0.0}
    estimated = {"calls": 0, "routed_usd": 0.0, "baseline_usd": 0.0}
    estimate_basis: dict[str, int] = defaultdict(int)
    measured_usage: dict[str, list[tuple[Usage, Optional[int]]]] = defaultdict(list)
    unmeasured: list[dict[str, Any]] = []
    unmeasured_causes: dict[str, int] = defaultdict(int)
    by_task: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    by_switch: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    by_day: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    by_surface: dict[str, int] = defaultdict(int)
    baseline_sources: dict[str, int] = defaultdict(int)
    unknown_baseline = changed = with_outcome = unpriced = 0

    # route_task advice a subagent then claimed counts once, as that subagent's route
    routes = [r for r in routes
              if r.get("tool_use_id") or r.get("request_id") is None or r["request_id"] not in claimed]
    for r in routes:
        by_surface[str(r.get("surface") or "?")] += 1
        baseline = r.get("baseline_model")
        routed = r.get("routed_model")
        baseline_sources[str(r.get("baseline_source") or "unknown")] += 1
        task = str(r.get("task_type") or "unknown")
        day = str(r.get("ts") or "")[:10]
        by_task[task]["calls"] += 1
        if not baseline:
            unknown_baseline += 1
            continue
        if baseline != routed:
            changed += 1
        switch = by_switch[(baseline, str(routed))]
        switch["calls"] += 1

        if r.get("estimated_cost_usd") is not None and r.get("baseline_estimated_cost_usd") is not None:
            estimated["calls"] += 1
            estimated["routed_usd"] += r["estimated_cost_usd"]
            estimated["baseline_usd"] += r["baseline_estimated_cost_usd"]
            estimate_basis[str(r.get("estimate_basis") or "unrecorded")] += 1

        outcome = outcomes.get((r.get("session_id"), r.get("tool_use_id"))) if r.get("tool_use_id") else None
        if outcome and not _usage(outcome) and outcome.get("agent_id"):
            # launched in the background: its usage came when it finished
            outcome = finished.get((outcome.get("session_id"), outcome["agent_id"])) or outcome
        usage = _usage(outcome) if outcome else None
        if outcome:
            with_outcome += 1
        if not usage:
            unmeasured.append(r)
            unmeasured_causes[_unmeasured_cause(r, outcome)] += 1
            continue
        ran = to_catalog_id(outcome.get("resolved_model"), catalog) or routed
        largest = outcome.get("max_request_prompt_tokens")
        largest = largest if isinstance(largest, int) else None
        actual = cost_of(usage, by_id.get(ran), largest)
        base = cost_of(usage, by_id.get(baseline), largest)
        if actual is None or base is None:
            unpriced += 1
            continue
        measured["calls"] += 1
        measured["actual_usd"] += actual
        measured["baseline_usd"] += base
        measured["tokens"] += sum(usage.values())
        measured_usage[task].append((usage, largest))
        for bucket in (by_task[task], switch, by_day[day]):
            bucket["measured_calls"] += 1
            bucket["actual_usd"] += actual
            bucket["baseline_usd"] += base

    for r in unmeasured:
        peers = measured_usage.get(str(r.get("task_type") or "unknown"), [])
        if len(peers) < PROJECTION_MIN_MEASURED:
            continue
        typical, largest = _median_call(peers)
        routed_cost = cost_of(typical, by_id.get(str(r.get("routed_model"))), largest)
        base_cost = cost_of(typical, by_id.get(str(r.get("baseline_model"))), largest)
        if routed_cost is None or base_cost is None:
            continue
        projected["calls"] += 1
        projected["routed_usd"] += routed_cost
        projected["baseline_usd"] += base_cost

    def finish(d: dict[str, Any], actual_key: str) -> dict[str, Any]:
        saved = d["baseline_usd"] - d[actual_key]
        return {**{k: round(v, 6) if isinstance(v, float) else v for k, v in d.items()},
                "saved_usd": round(saved, 6), "saved_pct": _pct(saved, d["baseline_usd"])}

    def rows(groups: dict[Any, dict[str, float]], label) -> list[dict[str, Any]]:
        out = []
        for key, g in groups.items():
            base, actual = g.get("baseline_usd", 0.0), g.get("actual_usd", 0.0)
            out.append({**label(key), "calls": int(g.get("calls", 0)),
                        "measured_calls": int(g.get("measured_calls", 0)),
                        "actual_usd": round(actual, 6), "baseline_usd": round(base, 6),
                        "saved_usd": round(base - actual, 6), "saved_pct": _pct(base - actual, base)})
        return sorted(out, key=lambda row: (-row["saved_usd"], -row["calls"]))

    routed_n = len(routes)
    notes = []
    if unknown_baseline:
        notes.append(f"{unknown_baseline} routed calls had no detectable baseline model and are not costed; "
                     "set ACTROUTER_BASELINE_MODEL (or the plugin's baseline_model) to include them.")
    if unpriced:
        notes.append(f"{unpriced} calls ran on or compared against a model without a catalog price.")
    if estimate_basis.get("unrecorded"):
        notes.append(f"{estimate_basis['unrecorded']} estimates predate token priors and price about 200 "
                     "tokens a call, far below a subagent's real work; read them as a direction, "
                     "not an amount.")
    for cause, n in sorted(unmeasured_causes.items(), key=lambda kv: -kv[1]):
        notes.append(f"{n} routed calls {UNMEASURED_CAUSES[cause]}; they are projected or estimated, "
                     "not measured.")

    return {
        "source": source,
        "period": {"from": first, "to": last},
        "calls": {
            "routed": routed_n,
            "skipped_explicit_model": skipped,
            "prompts_routed": prompts,
            "prompts_delegated": by_surface.get("prompt_hook", 0),
            "coverage_pct": _pct(routed_n, routed_n + skipped) if routed_n + skipped else None,
            "changed_model": changed,
            "with_outcome": with_outcome,
            "baseline_unknown": unknown_baseline,
            "unmeasured_by_cause": dict(sorted(unmeasured_causes.items())),
        },
        "measured": finish(measured, "actual_usd"),
        "projected": finish(projected, "routed_usd"),
        "estimated": {**finish(estimated, "routed_usd"), "basis": dict(sorted(estimate_basis.items()))},
        "by_task_type": rows(by_task, lambda k: {"task_type": k}),
        "by_switch": rows(by_switch, lambda k: {"baseline_model": k[0], "routed_model": k[1]}),
        "by_day": sorted(
            ({"day": d, "actual_usd": round(g["actual_usd"], 6), "baseline_usd": round(g["baseline_usd"], 6)}
             for d, g in by_day.items() if d),
            key=lambda row: row["day"],
        ),
        "by_surface": dict(sorted(by_surface.items())),
        "baseline_sources": dict(sorted(baseline_sources.items(), key=lambda kv: -kv[1])),
        "chat": _chat_summary(chat_turns),
        "notes": notes,
    }


def _chat_summary(turns: list[dict[str, Any]]) -> dict[str, Any]:
    """``actrouter-route chat`` turns: what the switch policy decided and what
    the turns cost. A switch decision's savings are the policy's estimate
    before the turn, not a measurement: the other model would have used
    other tokens."""
    reasons: dict[str, int] = defaultdict(int)
    out = {"turns": len(turns), "switched": 0, "would_switch_in_shadow": 0,
           "cost_computed_usd": 0.0, "cost_reported_usd": 0.0, "estimated_savings_of_switches_usd": 0.0}
    for t in turns:
        reasons[str(t.get("reason") or "?")] += 1
        if t.get("switched"):
            out["switched"] += 1
        if t.get("reason") == "switch":
            if t.get("shadow"):
                out["would_switch_in_shadow"] += 1
            saving = (t.get("gate") or {}).get("savings_usd")
            if isinstance(saving, (int, float)):
                out["estimated_savings_of_switches_usd"] += saving
        for key in ("cost_computed_usd", "cost_reported_usd"):
            if isinstance(t.get(key), (int, float)):
                out[key] += t[key]
    for key in ("cost_computed_usd", "cost_reported_usd", "estimated_savings_of_switches_usd"):
        out[key] = round(out[key], 6)
    out["reasons"] = dict(sorted(reasons.items(), key=lambda kv: -kv[1]))
    return out


def load(days: Optional[int], source: Optional[Path], catalog: list[dict[str, Any]]) -> dict[str, Any]:
    since = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)) if days else None
    target = source or ledger.path()
    return summarize(ledger.read(target, since=since), catalog, source=str(target or ""))


# --- text ------------------------------------------------------------------------


def _usd(value: Optional[float]) -> str:
    """$1,234.56, -$2.67, $0.0042 (sub-cent amounts keep 4 decimals), $0.00."""
    if value is None:
        return "-"
    sign = "-" if value < 0 else ""
    magnitude = abs(value)
    if magnitude == 0:
        return "$0.00"
    return f"{sign}${magnitude:,.4f}" if magnitude < 0.01 else f"{sign}${magnitude:,.2f}"


def _axis_usd(value: float, top: float) -> str:
    if top >= 10:
        return f"${value:,.0f}"
    return f"${value:,.2f}" if top >= 0.1 else f"${value:,.4f}"


def _day(ts: Optional[str]) -> str:
    return (ts or "-")[:10]


def format_text(s: dict[str, Any]) -> str:
    c, m, p, e = s["calls"], s["measured"], s["projected"], s["estimated"]
    lines = [
        f"ActRouter routing report  ({s['period']['from'] or '-'} .. {s['period']['to'] or '-'})",
        f"ledger: {s['source']}",
        "",
        f"routed calls        {c['routed']}  (model changed in {c['changed_model']})",
        f"left alone          {c['skipped_explicit_model']}  (model chosen explicitly)"
        + (f"; coverage {c['coverage_pct']}%" if c["coverage_pct"] is not None else ""),
        f"prompts routed      {c['prompts_routed']}  (another model fit; delegated to it {c['prompts_delegated']} times)",
        "",
        f"measured  ({m['calls']} calls with real usage)",
        f"  on routed models  {_usd(m['actual_usd'])}",
        f"  on baseline       {_usd(m['baseline_usd'])}",
        f"  saved             {_usd(m['saved_usd'])}" + (f"  ({m['saved_pct']}%)" if m["saved_pct"] is not None else ""),
        f"projected ({p['calls']} unmeasured calls, at their task type's median measured usage)",
        f"  saved             {_usd(p['saved_usd'])}" + (f"  ({p['saved_pct']}%)" if p["saved_pct"] is not None else ""),
        f"estimated ({e['calls']} calls, before the fact"
        + (f"; basis {', '.join(f'{k} {v}' for k, v in e['basis'].items())}" if e["basis"] else "") + ")",
        f"  saved             {_usd(e['saved_usd'])}" + (f"  ({e['saved_pct']}%)" if e["saved_pct"] is not None else ""),
    ]
    if s["by_task_type"]:
        lines += ["", f"{'task type':<18}{'calls':>6}{'measured':>10}{'saved':>12}"]
        for row in s["by_task_type"][:10]:
            lines.append(f"{row['task_type']:<18}{row['calls']:>6}{row['measured_calls']:>10}{_usd(row['saved_usd']):>12}")
    if s["by_switch"]:
        lines += ["", "most common switches (baseline -> routed):"]
        for row in sorted(s["by_switch"], key=lambda r: -r["calls"])[:6]:
            lines.append(f"  {row['baseline_model']} -> {row['routed_model']}: {row['calls']} calls, saved {_usd(row['saved_usd'])}")
    chat = s.get("chat") or {}
    if chat.get("turns"):
        lines += [
            "",
            f"chat turns          {chat['turns']}  (switched {chat['switched']}; "
            f"would have switched in shadow {chat['would_switch_in_shadow']})",
            f"  cost from tokens  {_usd(chat['cost_computed_usd'])}  (main session)"
            + (f"; reported by Claude Code {_usd(chat['cost_reported_usd'])}" if chat["cost_reported_usd"] else ""),
            f"  switch savings    {_usd(chat['estimated_savings_of_switches_usd'])}  (estimated before each turn)",
            "  decisions         " + ", ".join(f"{k} {v}" for k, v in chat["reasons"].items()),
        ]
    if s["notes"]:
        lines += [""] + [f"note: {n}" for n in s["notes"]]
    return "\n".join(lines)


# --- HTML ------------------------------------------------------------------------


def _chart(days: list[dict[str, Any]]) -> str:
    """Daily cost on the baseline vs on the routed models, as inline SVG."""
    if not days:
        return '<p class="empty">No measured calls yet. Outcomes are recorded by the Claude Code PostToolUse hook.</p>'
    days = days[-31:]
    top = max(max(d["baseline_usd"], d["actual_usd"]) for d in days) or 1.0
    width, height, pad_l, pad_b, pad_t = 720, 230, 60, 28, 14
    plot = height - pad_b - pad_t
    slot = (width - pad_l) / len(days)
    bar = max(2.0, min(14.0, slot / 2.6))
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-labelledby="chart-title" class="chart">',
        '<title id="chart-title">Daily cost on the baseline model versus the routed model</title>',
    ]
    for i in range(5):
        y = pad_t + plot * (1 - i / 4)
        parts.append(f'<line x1="{pad_l}" x2="{width}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{pad_l - 6}" y="{y + 4:.1f}" class="axis" text-anchor="end">{html.escape(_axis_usd(top * i / 4, top))}</text>')
    for i, d in enumerate(days):
        x = pad_l + i * slot + slot / 2
        for value, cls, dx in ((d["baseline_usd"], "base", -bar), (d["actual_usd"], "actual", 0.0)):
            h = plot * value / top
            parts.append(
                f'<rect x="{x + dx:.1f}" y="{pad_t + plot - h:.1f}" width="{bar:.1f}" height="{h:.1f}" class="{cls}">'
                f'<title>{html.escape(d["day"])}: {"baseline" if cls == "base" else "routed"} {html.escape(_usd(value))}</title></rect>'
            )
        if len(days) <= 14 or i % max(1, len(days) // 7) == 0:
            parts.append(f'<text x="{x:.1f}" y="{height - 8}" class="axis" text-anchor="middle">{html.escape(d["day"][5:])}</text>')
    parts.append("</svg>")
    return '<div class="chart-wrap">' + "".join(parts) + "</div>"


def _numeric(key: str) -> bool:
    return key.endswith(("_usd", "_pct", "calls"))


def _table(rows: list[dict[str, Any]], columns: list[tuple[str, str]], empty: str) -> str:
    if not rows:
        return f'<p class="empty">{html.escape(empty)}</p>'
    head = "".join(
        f'<th scope="col"{" class=\"num\"" if _numeric(key) else ""}>{html.escape(label)}</th>'
        for key, label in columns
    )
    body = []
    for row in rows:
        cells = []
        for key, _ in columns:
            value = row.get(key)
            if key.endswith("_usd"):
                text = _usd(value)
            elif key.endswith("_pct"):
                text = "-" if value is None else f"{value}%"
            else:
                text = "-" if value is None else str(value)
            negative = key.startswith("saved") and isinstance(value, (int, float)) and value < 0
            cls = f' class="num{" neg" if negative else ""}"' if _numeric(key) else ""
            cells.append(f"<td{cls}>{html.escape(text)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def format_html(s: dict[str, Any]) -> str:
    c, m, p, e = s["calls"], s["measured"], s["projected"], s["estimated"]
    tiles = [
        ("Saved (measured)", _usd(m["saved_usd"]), f"{m['saved_pct']}% lower cost" if m["saved_pct"] is not None else f"{m['calls']} calls with real usage"),
        ("Saved (projected)", _usd(p["saved_usd"]), f"{p['calls']} unmeasured calls, at measured medians"),
        ("Saved (estimated)", _usd(e["saved_usd"]), f"{e['calls']} routed calls, before the fact"),
        ("Routed calls", str(c["routed"]), f"model changed in {c['changed_model']}"),
        ("Coverage", f"{c['coverage_pct']}%" if c["coverage_pct"] is not None else "-", f"{c['skipped_explicit_model']} left on an explicit model"),
    ]
    tile_html = "".join(
        f'<div class="tile"><div class="label">{html.escape(a)}</div><div class="value">{html.escape(b)}</div><div class="sub">{html.escape(cc)}</div></div>'
        for a, b, cc in tiles
    )
    notes = "".join(f"<li>{html.escape(n)}</li>" for n in s["notes"])
    sources = ", ".join(f"{html.escape(k)} ({v})" for k, v in s["baseline_sources"].items()) or "-"
    period = f"{_day(s['period']['from'])} to {_day(s['period']['to'])}"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ActRouter Savings</title>
<style>
:root {{ --bg:#f7f7f5; --panel:#ffffff; --text:#1d1d1b; --muted:#6b6b66; --line:#e3e2dd;
  --base:#b9b6ad; --actual:#2f6f5e; --accent:#2f6f5e; --neg:#a3452f; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#161615; --panel:#1f1f1d; --text:#ecebe6;
  --muted:#a09f99; --line:#33332f; --base:#5c5a54; --actual:#6fb59f; --accent:#6fb59f; --neg:#e08a72; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--text);
  font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }}
main {{ max-width:960px; margin:0 auto; padding:24px 16px 48px; }}
h1 {{ font-size:22px; margin:0 0 4px; }} h2 {{ font-size:16px; margin:32px 0 10px; }}
.meta {{ color:var(--muted); font-size:13px; margin:0 0 20px; overflow-wrap:anywhere; }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; }}
.tile {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px 16px; }}
.label {{ color:var(--muted); font-size:13px; }} .value {{ font-size:26px; font-weight:650; margin:2px 0; }}
.sub {{ color:var(--muted); font-size:13px; }}
.panel {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px 16px; }}
.chart-wrap {{ overflow-x:auto; }} .chart {{ width:100%; min-width:560px; height:auto; display:block; }}
.grid {{ stroke:var(--line); }}
.axis {{ fill:var(--muted); font-size:11px; }} .base {{ fill:var(--base); }} .actual {{ fill:var(--actual); }}
.legend {{ display:flex; gap:16px; font-size:13px; color:var(--muted); margin-top:6px; }}
.swatch {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:6px; vertical-align:-1px; }}
.scroll {{ overflow-x:auto; }} table {{ width:100%; border-collapse:collapse; font-size:14px; }}
th, td {{ text-align:left; padding:7px 8px; border-bottom:1px solid var(--line); white-space:nowrap; }}
th {{ color:var(--muted); font-weight:500; }} td.num, th.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
td.neg {{ color:var(--neg); }}
.empty {{ color:var(--muted); }} ul {{ padding-left:18px; }} li {{ margin:4px 0; }}
.method {{ color:var(--muted); font-size:13px; }}
</style>
</head>
<body>
<main>
<h1>ActRouter savings</h1>
<p class="meta">{html.escape(period)}<br>ledger: {html.escape(s['source'])}</p>
<section class="tiles">{tile_html}</section>

<h2>Daily cost: baseline vs routed (measured)</h2>
<div class="panel">{_chart(s['by_day'])}
<div class="legend"><span><span class="swatch" style="background:var(--base)"></span>baseline model</span>
<span><span class="swatch" style="background:var(--actual)"></span>routed model</span></div></div>

<h2>By task type</h2>
<div class="panel">{_table(s['by_task_type'], [("task_type", "Task type"), ("calls", "Calls"), ("measured_calls", "Measured"), ("baseline_usd", "Baseline"), ("actual_usd", "Routed"), ("saved_usd", "Saved"), ("saved_pct", "Saved %")], "No routed calls yet.")}</div>

<h2>Model switches</h2>
<div class="panel">{_table(s['by_switch'], [("baseline_model", "Baseline"), ("routed_model", "Routed to"), ("calls", "Calls"), ("measured_calls", "Measured"), ("saved_usd", "Saved"), ("saved_pct", "Saved %")], "No routed calls with a known baseline yet.")}</div>

<h2>How this is measured</h2>
<div class="panel method"><ul>
<li><b>Measured</b>: each routed subagent's real token usage (from Claude Code) priced on the model that ran and on the baseline model. Same tokens on both sides, so it shows the price effect of the switch; another model would not use exactly the same tokens.</li>
<li><b>Projected</b>: routed calls with no recorded outcome, priced at the median measured usage of the same task type (once {PROJECTION_MIN_MEASURED} of its calls were measured).</li>
<li><b>Estimated</b>: the router's before-the-fact estimate from the task's token estimate (the caller's, or the task type's typical agent work), for every routed call.</li>
<li>Baseline: the model that would have run without ActRouter. Sources here: {sources}.</li>
<li>Prices: the ActRouter catalog, including cached-token rates and long-context tiers. Cache writes are priced at the input rate.</li>
{notes}
</ul></div>
</main>
</body>
</html>
"""
